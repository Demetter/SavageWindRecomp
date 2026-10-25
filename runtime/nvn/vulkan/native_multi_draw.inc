// Guarded native graphics pilot. The producer resolves resources, the existing
// descriptor worker snapshots constants, and the recorder groups adjacent draws.
// The queue flushes before every other command and at every drain/submission.
typedef struct {
    VkCommandBuffer cb;
    const Pipe *pipe; // immutable pipeline/vertex layout; do not copy it per draw
    DrawJob *job;
    NativeDrawSlot slot;
    VkBuffer vertex[17], index;
    VkDeviceSize offsets[17], index_offset;
    VkIndexType index_type;
    uint32_t count, instances, base_instance;
    int32_t base_vertex;
    uint64_t seq;
} NativeDrawPacket;

static NativeDrawSlot native_draw_slot(NativeProgram *p,uint64_t draw_seq) {
    static NativeDrawSlot next;
    static uint64_t seq;
    if(seq!=draw_seq||!next.span.buf||next.index==NATIVE_DRAW_CAP) {
        seq=draw_seq;
        uint64_t align=g_vk.props.limits.minStorageBufferOffsetAlignment;if(align<16)align=16;
        next.span=vk_transient(sizeof(NativeDrawUpload),align);next.index=0;
        VkDescriptorBufferInfo bi={next.span.buf,next.span.off,sizeof(((NativeDrawUpload*)0)->records)};
        VkWriteDescriptorSet w={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstBinding=0,.descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi};
        next.table=set_finish(p->table_layout,&w,1);
    }
    NativeDrawSlot slot=next;next.index++;return slot;
}

static VkDeviceAddress native_buffer_address(VkBuffer buffer) {
    static __thread struct { VkBuffer buffer; VkDeviceAddress address; } memo[32];
    unsigned slot=(unsigned)(((uintptr_t)buffer>>4)*0x9e3779b1u)&31;
    if(memo[slot].buffer!=buffer) {
        VkBufferDeviceAddressInfo ai={.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer=buffer};
        memo[slot].buffer=buffer;memo[slot].address=vkGetBufferDeviceAddress(g_vk.dev,&ai);
    }
    return memo[slot].address;
}

static void native_draw_constants_fill(DrawJob *job) {
    NativeProgram *p=job->pr->native_program;
    NativeDrawUpload *upload=job->native_slot.span.ptr;
    NativeDrawConstants *record=&upload->records[job->native_slot.index];
    // Write-combined upload memory must never be read back for deduplication.
    NativeDrawConstants local={0};
    for(unsigned stage=0;stage<2;stage++)for(uint32_t mask=p->constant_mask[stage];mask;mask&=mask-1) {
        unsigned k=(unsigned)__builtin_ctz(mask);
        VkDescriptorBufferInfo bi=cbuf_info((int)stage,stage?job->pr->fi:job->pr->vi,(int)k,0,0,0);
        VkDeviceAddress address=native_buffer_address(bi.buffer);
        if(!address||bi.range==VK_WHOLE_SIZE||bi.range/4>UINT32_MAX)FATAL("native draw: invalid constant upload range\n");
        local.constants[stage*18+k]=(NativeDrawConstant){address+bi.offset,(uint32_t)(bi.range/4),0};
    }
    memcpy(record,&local,sizeof local);
    uploads_dirty=true; // host_upload_publish runs on THIS worker before job-ready
}

static struct {
    NativeDrawPacket first,last;
    VkDescriptorSet sets[2];
    uint32_t dyn[32],ndyn;
    VkDeviceSize index_binding;
    VkDrawIndexedIndirectCommand commands[NATIVE_DRAW_CAP];
    unsigned count;
} native_batch;
static _Atomic uint64_t native_draws_recorded,native_batches_recorded;
static void native_draw_frame(uint64_t frame) {
    if(!g_vk.native_multi_draw)return;
    static int stats=-1;if(stats<0)stats=getenv("SAVAGEWIND_VK_NATIVE_MULTI_DRAW_STATS")!=NULL;
    if(!stats)return;
    uint64_t draws=atomic_exchange_explicit(&native_draws_recorded,0,memory_order_relaxed);
    uint64_t batches=atomic_exchange_explicit(&native_batches_recorded,0,memory_order_relaxed);
    if(draws)LOG("[native-multi-draw] frame %llu draws %llu batches %llu\n",(unsigned long long)frame,(unsigned long long)draws,(unsigned long long)batches);
}

static void native_bind_vertices(const NativeDrawPacket *p) {
    uint32_t streams=p->pipe->k.s.used_streams;
    for(unsigned s=0;s<17;) {
        if(!(streams&(1u<<s))){s++;continue;}
        unsigned first=s++;while(s<17&&(streams&(1u<<s)))s++;
        vkCmdBindVertexBuffers(p->cb,first,s-first,p->vertex+first,p->offsets+first);
    }
}

void vk_native_batch_flush(void) {
    if(!native_batch.count)return;
    NativeDrawPacket *first=&native_batch.first,*last=&native_batch.last;
    NativeDrawUpload *upload=first->slot.span.ptr;
    memcpy(upload->commands+first->slot.index,native_batch.commands,native_batch.count*sizeof native_batch.commands[0]);
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("sfence" ::: "memory");
#else
    atomic_thread_fence(memory_order_seq_cst);
#endif
    VkDescriptorSet sets[3]={native_batch.sets[0],native_batch.sets[1],first->slot.table};
    VkPipelineLayout layout=first->pipe->k.prog->native_program->program.layout;
    vkCmdBindDescriptorSets(first->cb,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,3,sets,native_batch.ndyn,native_batch.dyn);
    vkCmdPushConstants(first->cb,layout,VK_SHADER_STAGE_VERTEX_BIT,0,4,&first->slot.index);
    native_bind_vertices(first);
    vkCmdBindIndexBuffer(first->cb,first->index,native_batch.index_binding,first->index_type);
    VkDeviceSize indirect=first->slot.span.off+offsetof(NativeDrawUpload,commands)+first->slot.index*sizeof(VkDrawIndexedIndirectCommand);
    vkCmdDrawIndexedIndirect(first->cb,first->slot.span.buf,indirect,native_batch.count,sizeof(VkDrawIndexedIndirectCommand));
    // Preserve the binding state the producer's dc cache believes is current.
    native_bind_vertices(last);
    vkCmdBindIndexBuffer(last->cb,last->index,last->index_offset,last->index_type);
    atomic_fetch_add_explicit(&native_draws_recorded,native_batch.count,memory_order_relaxed);
    atomic_fetch_add_explicit(&native_batches_recorded,1,memory_order_relaxed);
    native_batch.count=0;
}

void vk_native_batch_replay(void *arg) {
    NativeDrawPacket *p=arg;DrawJob *job=p->job;dj_wait_ready(job);
    static unsigned limit;
    if(!limit) {
        const char *e=getenv("SAVAGEWIND_VK_NATIVE_BATCH_LIMIT");int n=e?atoi(e):NATIVE_DRAW_CAP;
        limit=n>0&&n<=NATIVE_DRAW_CAP?(unsigned)n:NATIVE_DRAW_CAP;
    }
    NativeDrawPacket *first=&native_batch.first;
    bool same=native_batch.count&&native_batch.count<limit&&first->cb==p->cb&&first->seq==p->seq&&
        first->pipe==p->pipe&&
        first->index==p->index&&first->index_type==p->index_type&&first->slot.span.buf==p->slot.span.buf&&
        first->slot.span.off==p->slot.span.off&&first->slot.table==p->slot.table&&
        first->slot.index+native_batch.count==p->slot.index&&
        native_batch.sets[0]==job->sets[0]&&native_batch.sets[1]==job->sets[1]&&native_batch.ndyn==(uint32_t)job->ndyn&&
        !memcmp(native_batch.dyn,job->dyn,(size_t)job->ndyn*sizeof job->dyn[0]);
    const PipeState *state=&p->pipe->k.s;
    for(unsigned s=0;s<17&&same;s++)if(state->used_streams&(1u<<s))same=first->vertex[s]==p->vertex[s];
    int32_t base=p->base_vertex;int64_t delta=0;uint64_t binding=p->index_offset;uint32_t shift=0,index=0;
    if(same)same=native_draw_vertex_rebase(state->used_streams,first->offsets,p->offsets,state->stride,state->divisor,p->base_vertex,&base,&delta)&&
        native_draw_index_rebase(native_batch.index_binding,p->index_offset,p->index_type==VK_INDEX_TYPE_UINT16?2:4,&binding,&shift,&index);
    if(same&&shift)for(unsigned i=0;i<native_batch.count;i++)if(native_batch.commands[i].firstIndex>UINT32_MAX-shift){same=false;break;}
    if(!same) {
        vk_native_batch_flush();native_batch.first=*p;native_batch.index_binding=p->index_offset;
        native_batch.sets[0]=job->sets[0];native_batch.sets[1]=job->sets[1];base=p->base_vertex;index=0;
        native_batch.ndyn=(uint32_t)job->ndyn;memcpy(native_batch.dyn,job->dyn,(size_t)job->ndyn*sizeof job->dyn[0]);
    } else {
        for(unsigned i=0;i<native_batch.count;i++)native_batch.commands[i].firstIndex+=shift;
        native_batch.index_binding=binding;
    }
    native_batch.commands[native_batch.count++]=(VkDrawIndexedIndirectCommand){p->count,p->instances,index,base,p->base_instance};
    native_batch.last=*p;
    atomic_store_explicit(&job->state,0,memory_order_release);
    if(vkq_address_wait)vkq_wake_address(&job->state,true);
}
