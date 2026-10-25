// A1 diagnostic only. Queue order owns the shadow; normal guest fences remain.
// A sparse word-dirty bitmap writes back exactly the words touched by shaders,
// not whole allocations (which can contain unrelated CPU-owned bytes).
#define BC_MAX 2048
enum { BC_UNKNOWN_READ=1, BC_UNKNOWN_WRITE=2, BC_UNSUPPORTED=4, BC_CAPACITY=8,
       BC_LIVE_VERTEX=16, BC_INPUT_COPY=32, BC_CPU_UPDATE=64, BC_BAD_RANGE=128,
       BC_LIVE_GRAPHICS=256, BC_PENDING_AUDIT=512 };
static bool bc_active, bc_writes, bc_have_frame, bc_original_writer;
static uint64_t bc_frame, bc_generation, bc_first_ns, bc_copy_seq, bc_vertex_seq;
static uint64_t bc_copy_bytes, bc_aliases, bc_dispatches, bc_draws, bc_rb_bytes, bc_rb_skipped;
static unsigned bc_reasons, bc_n;
static int bc_audit;
static uint64_t bc_map_version, bc_updates, bc_update_drains;
// A1c producer CPU wall spans. Write-back/barrier recording can be nested in a
// uniform drain; the log explicitly retains both inclusive and exclusive time.
typedef struct {
    uint64_t uniform_n,uniform_ns,uniform_wb_ns,uniform_inpass,other_drains;
    uint64_t map_n,map_ns,snapshot_n,snapshot_ns;
    uint64_t wb_n,wb_dispatches,wb_ns,barriers,barrier_ns;
    uint64_t view_n,view_ns,update_gpu_n,update_gpu_bytes;
} BcCost;
static BcCost bc_cost;
static void bc_barrier(VkCommandBuffer cb) {
    uint64_t t=plat_now_ns();vk_full_barrier(cb);
    bc_cost.barriers++;bc_cost.barrier_ns+=plat_now_ns()-t;
}
typedef struct { const ShaderInfo *si; uint64_t id; int stage; bool proved, rewritten; } BcShader;
static BcShader bc_shaders[BC_SHADER_COUNT];
static unsigned bc_nshaders;
static uint32_t bc_gfx_draws[BC_SHADER_COUNT], bc_gfx_unbounded[BC_SHADER_COUNT];
static VkDescriptorBufferInfo bc_graphics_map[2];
static __thread const VkDescriptorBufferInfo *t_bc_maps;
static bool bc_readback_owned(uint64_t fs, uint64_t bytes) {
    return bc_active && fs == 0x6cb8919fb3be8cd0ull && bytes == 1048576;
}
static void bc_readback_note(uint64_t bytes, bool gpu_only) {
    if (!vk_buffer_cache_configured()) return;
    bc_rb_bytes += bytes;
    if (gpu_only) bc_rb_skipped += bytes;
}
static struct { uint64_t lo, hi; } bc_ranges[BC_MAX]; // byte offsets, sorted/disjoint
// CPU ownership snapshots are immutable until the job's GPU submission retires.
// A worker must never consult the producer's expanding current-frame map.
typedef struct BcView {
    struct BcView *next;
    uint64_t seq;
    unsigned n;
    bool active;
    struct {uint64_t lo,hi;} ranges[];
} BcView;
static BcView *bc_views;
static __thread const BcView *t_bc_view;
static BcView *bc_job_view(void) {
    if(!vk_buffer_cache_configured())return NULL;
    static uint64_t version,seq;
    static BcView *last;
    uint64_t pending=vk_pending_seq();
    if(last&&version==bc_map_version&&seq==pending)return last;
    uint64_t begin=plat_now_ns();unsigned n=bc_active?bc_n:0;
    BcView *v=malloc(sizeof *v+n*sizeof v->ranges[0]);if(!v)FATAL("A1c ownership view allocation failed\n");
    *v=(BcView){.next=bc_views,.seq=pending,.n=n,.active=bc_active};
    memcpy(v->ranges,bc_ranges,n*sizeof v->ranges[0]);bc_views=v;
    version=bc_map_version;seq=pending;last=v;
    bc_cost.view_n++;bc_cost.view_ns+=plat_now_ns()-begin;return v;
}
static pthread_mutex_t bc_lock=PTHREAD_MUTEX_INITIALIZER;
static VkBuffer bc_buffer, bc_dirty;
static VmaAllocation bc_mem, bc_dirty_mem;
static uint64_t bc_size, bc_dirty_size;
#define BC_AUDIT_BYTES (BC_SHADER_COUNT*BC_AUDIT_WORDS*4u)
static VkDescriptorBufferInfo bc_map;
static VkPipeline bc_wb_pipe;
static VkPipelineLayout bc_wb_layout;
static VkDescriptorSetLayout bc_wb_set;
typedef struct { VkBuffer buf; VmaAllocation mem; uint32_t *ptr; uint64_t seq,frame,generation,copy_bytes,aliases,dispatches,draws,rb_bytes,rb_skipped,updates,drains; unsigned reasons; bool on; BcCost cost; uint32_t gfx_draws[BC_SHADER_COUNT],gfx_unbounded[BC_SHADER_COUNT]; } BcResult;
static BcResult bc_results[64];
static unsigned bc_result_n;
static int bc_mode, bc_ab;
static pthread_once_t bc_once=PTHREAD_ONCE_INIT;
static void bc_config(void) {
    const char *e=getenv("SAVAGEWIND_VK_BUFFER_CACHE");bc_mode=e?atoi(e):0;
    e=getenv("SAVAGEWIND_VK_BUFFER_CACHE_AB");bc_ab=e?atoi(e):0;
    e=getenv("SAVAGEWIND_VK_BUFFER_CACHE_AUDIT");bc_audit=e?atoi(e):0;
    if (bc_mode || bc_ab) {
        if (early_fence() || renderer_safe_overlap_enabled() || perf_survey_mode || hc_on() || co_on())
            FATAL("A1 requires normal fences and no other GPU experiment\n");
        LOG("[a1] diagnostic owned buffers; normal fences; no early release; AB=%d\n",bc_ab);
    }
}
bool vk_buffer_cache_configured(void) { pthread_once(&bc_once,bc_config);return bc_mode||bc_ab; }
static bool bc_covers_locked(uint64_t lo,uint64_t hi) {
    unsigned a=0,b=bc_n;
    while(a<b){unsigned m=(a+b)/2;if(bc_ranges[m].hi<=lo)a=m+1;else b=m;}
    return a<bc_n&&bc_ranges[a].lo<=lo&&hi<=bc_ranges[a].hi;
}
static bool bc_direct(uint64_t addr,uint64_t size,VkBuffer *buf,VkDeviceSize *off) {
    if(!size||addr<HEAP_BASE||addr-HEAP_BASE>=bc_size||size>bc_size-(addr-HEAP_BASE))return false;
    uint64_t lo=addr-HEAP_BASE,hi=lo+size;bool ok=false;
    if(t_worker>=0){
        const BcView *v=t_bc_view;if(!v||!v->active)return false;
        unsigned a=0,b=v->n;while(a<b){unsigned m=(a+b)/2;if(v->ranges[m].hi<=lo)a=m+1;else b=m;}
        ok=a<v->n&&v->ranges[a].lo<=lo&&hi<=v->ranges[a].hi;
    }else{if(!bc_active)return false;ok=bc_covers_locked(lo,hi);}
    if(ok){*buf=bc_buffer;*off=lo;if(t_worker<0)bc_aliases++;}return ok;
}
static void bc_make_writeback(void) {
    if(bc_wb_pipe)return;
    const char *src="#version 460\nlayout(local_size_x=64) in;\n"
        "layout(std430,binding=0) readonly buffer Owned {uint w[];} owned_;\n"
        "layout(std430,binding=1) buffer Dirty {uint w[];} dirty_;\n"
        "layout(std430,binding=2) buffer Guest {uint w[];} guest_;\n"
        "layout(std430,binding=3) readonly buffer Ranges {uvec4 r[];} ranges_;\n"
        "layout(push_constant) uniform P {uint n;uint count;} p;\n"
        "void main(){uint x=gl_GlobalInvocationID.x;if(x>=p.count)return;uint lo=0u,hi=p.n;"
        "while(lo<hi){uint m=(lo+hi)>>1u;if(ranges_.r[m].z<=x)lo=m+1u;else hi=m;}"
        "uvec4 r=ranges_.r[lo];uint k=r.x+x-(r.z-r.y);"
        "uint bits=atomicExchange(dirty_.w[4u+k],0u);while(bits!=0u){uint b=uint(findLSB(bits));"
        "uint i=k*32u+b;guest_.w[i]=owned_.w[i];bits&=bits-1u;}}\n";
    size_t nw=0;uint32_t *spv=vk_glsl_to_spirv(src,STAGE_COMPUTE,"a1-word-writeback",&nw);
    if(!spv)FATAL("A1 write-back shader failed\n");VkShaderModule mod=vk_module(spv,nw);free(spv);
    VkBindingInfo binds[4]={{0,0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},{0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},{0,2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},{0,3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}};
    bc_wb_set=vk_set_layout(binds,4,0,VK_SHADER_STAGE_COMPUTE_BIT);
    VkPushConstantRange push={VK_SHADER_STAGE_COMPUTE_BIT,0,8};
    VkPipelineLayoutCreateInfo li={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&bc_wb_set,.pushConstantRangeCount=1,.pPushConstantRanges=&push};
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev,&li,NULL,&bc_wb_layout));
    VkComputePipelineCreateInfo ci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.layout=bc_wb_layout,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=mod,.pName="main"}};
    VK_CHECK(vkCreateComputePipelines(g_vk.dev,VK_NULL_HANDLE,1,&ci,NULL,&bc_wb_pipe));vkDestroyShaderModule(g_vk.dev,mod,NULL);
}
static void bc_buffers(void) {
    if(bc_buffer)return;
    fx_quiet=true;gmem_info();fx_quiet=false;
    if(!gmem_buf)FATAL("A1 cannot import the guest heap\n");
    bc_size=gmem_size;bc_dirty_size=(16+((bc_size+127)/128)*4+255)&~255ull;
    VkBufferCreateInfo bi={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bc_size,
        .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai={.usage=VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VK_CHECK(vmaCreateBuffer(g_vk.vma,&bi,&ai,&bc_buffer,&bc_mem,NULL));bi.size=bc_dirty_size+BC_AUDIT_BYTES;
    VK_CHECK(vmaCreateBuffer(g_vk.vma,&bi,&ai,&bc_dirty,&bc_dirty_mem,NULL));
    void *zero=calloc(1,(size_t)bi.size);if(!zero)FATAL("A1 dirty init allocation\n");
    VkSpan z=vk_transient_copy(zero,bi.size,256);free(zero);
    VkBufferCopy copy={z.off,0,bi.size};VkCommandBuffer cb=vk_cmd();vkCmdCopyBuffer(cb,z.buf,bc_dirty,1,&copy);vk_full_barrier(cb);
    bc_make_writeback();LOG("[a1] owned heap %llu MiB; dirty map %llu bytes\n",(unsigned long long)(bc_size>>20),(unsigned long long)bc_dirty_size);
}
static void bc_before_submit(void) {
    if(!bc_active||!bc_writes)return;
    uint64_t begin=plat_now_ns();bc_cost.wb_n++;
    VkCommandBuffer cb=vk_cmd();bc_barrier(cb); // earlier shader writes -> batch reads
    uint32_t ranges[BC_MAX][4],n=0,total=0;
    for(unsigned i=0;i<bc_n;i++){
        uint32_t first=(uint32_t)(bc_ranges[i].lo/128),end=(uint32_t)((bc_ranges[i].hi+127)/128);
        if(n&&first<=ranges[n-1][0]+ranges[n-1][1]){
            uint32_t old_end=ranges[n-1][0]+ranges[n-1][1];
            if(end>old_end){ranges[n-1][1]+=end-old_end;total+=end-old_end;ranges[n-1][2]=total;}
        }else{total+=end-first;ranges[n][0]=first;ranges[n][1]=end-first;ranges[n][2]=total;ranges[n][3]=0;n++;}
    }
    if(n){
        VkSpan span=vk_transient_copy(ranges,n*sizeof ranges[0],256);
        VkDescriptorSet ds=vk_alloc_set(bc_wb_set);
        VkDescriptorBufferInfo bi[4]={{bc_buffer,0,bc_size},{bc_dirty,0,bc_dirty_size},{gmem_buf,0,gmem_size},{span.buf,span.off,n*sizeof ranges[0]}};
        VkWriteDescriptorSet w[4];for(unsigned i=0;i<4;i++)w[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
        vkUpdateDescriptorSets(g_vk.dev,4,w,0,NULL);vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,bc_wb_pipe);
        vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,bc_wb_layout,0,1,&ds,0,NULL);
        uint32_t push[2]={n,total};vkCmdPushConstants(cb,bc_wb_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,push);
        vkCmdDispatch(cb,(total+63)/64,1,1);bc_cost.wb_dispatches++;
    }
    // One output barrier for the whole batch, including host visibility after
    // the ordinary guest fence; not a barrier per owned interval.
    uint64_t t=plat_now_ns();
    VkMemoryBarrier2 mb={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,.srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask=VK_ACCESS_2_SHADER_WRITE_BIT,.dstStageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT|VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask=VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT|VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di={.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,.memoryBarrierCount=1,.pMemoryBarriers=&mb};
    vkCmdPipelineBarrier2(cb,&di);bc_cost.barriers++;bc_cost.barrier_ns+=plat_now_ns()-t;
    fx_mark_r(FXR_GMEM);bc_writes=false;bc_cost.wb_ns+=plat_now_ns()-begin;
}
static void bc_collect(void) {
    uint64_t done=vk_completed_seq();unsigned keep=0;
    BcView **v=&bc_views;while(*v){if((*v)->seq<=done){BcView *old=*v;*v=old->next;free(old);}else v=&(*v)->next;}
    for(unsigned i=0;i<bc_result_n;i++){
        BcResult *r=&bc_results[i];if(r->seq>done){bc_results[keep++]=*r;continue;}
        VK_CHECK(vmaInvalidateAllocation(g_vk.vma,r->mem,0,VK_WHOLE_SIZE));
        unsigned why=r->reasons|r->ptr[0];
        TRACE_LOG("[a1-frame] frame %llu gen %llu on %u seq %llu reasons %u eligible %u copy_bytes %llu writeback_bytes %u read_misses %u write_misses %u aliases %llu dispatches %llu draws %llu readback_bytes %llu gpu_only_bytes %llu\n",
            (unsigned long long)r->frame,(unsigned long long)r->generation,r->on,(unsigned long long)r->seq,why,r->on&&!why,
            (unsigned long long)r->copy_bytes,r->ptr[1],r->ptr[2],r->ptr[3],(unsigned long long)r->aliases,
            (unsigned long long)r->dispatches,(unsigned long long)r->draws,(unsigned long long)r->rb_bytes,(unsigned long long)r->rb_skipped);
        TRACE_LOG("[a1b-updates] frame %llu on %u changed_overlaps %llu drains %llu rejected %u\n",(unsigned long long)r->frame,r->on,(unsigned long long)r->updates,(unsigned long long)r->drains,!!(why&BC_CPU_UPDATE));
        BcCost *c=&r->cost;
        TRACE_LOG("[a1c-cost] frame %llu on %u uniform_n %llu uniform_ns %llu uniform_wb_ns %llu uniform_inpass %llu other_drains %llu map_n %llu map_ns %llu snapshot_n %llu snapshot_ns %llu wb_n %llu wb_dispatches %llu wb_ns %llu barriers %llu barrier_ns %llu view_n %llu view_ns %llu update_gpu_n %llu update_gpu_bytes %llu\n",
            (unsigned long long)r->frame,r->on,(unsigned long long)c->uniform_n,(unsigned long long)c->uniform_ns,(unsigned long long)c->uniform_wb_ns,(unsigned long long)c->uniform_inpass,(unsigned long long)c->other_drains,
            (unsigned long long)c->map_n,(unsigned long long)c->map_ns,(unsigned long long)c->snapshot_n,(unsigned long long)c->snapshot_ns,
            (unsigned long long)c->wb_n,(unsigned long long)c->wb_dispatches,(unsigned long long)c->wb_ns,(unsigned long long)c->barriers,(unsigned long long)c->barrier_ns,(unsigned long long)c->view_n,(unsigned long long)c->view_ns,(unsigned long long)c->update_gpu_n,(unsigned long long)c->update_gpu_bytes);
        for(unsigned k=0;k<bc_nshaders;k++) if(r->gfx_draws[k]) {
            uint32_t *a=bc_audit?r->ptr+4+k*BC_AUDIT_WORDS:NULL;
            char samples[400]={0};unsigned at=0;
            if(a)for(unsigned j=0;j<16&&j<a[0];j++)at+=(unsigned)snprintf(samples+at,sizeof samples-at,"%s%llx",j?",":"",(unsigned long long)(HEAP_BASE+4ull*a[8+j]));
            TRACE_LOG("[a1b-shader] frame %llu on %u id %016llx stage %d proved %u draws %u unbounded %u audit %u reads %u writes %u min %llx max %llx outside_map %u samples %s\n",
                (unsigned long long)r->frame,r->on,(unsigned long long)bc_shaders[k].id,bc_shaders[k].stage,bc_shaders[k].proved,r->gfx_draws[k],r->gfx_unbounded[k],bc_audit!=0,
                a?a[0]:0,a?a[1]:0,(unsigned long long)(a&&(a[0]||a[1])?HEAP_BASE+4ull*(uint32_t)~a[2]:0),
                (unsigned long long)(a&&(a[0]||a[1])?HEAP_BASE+4ull*a[3]:0),a?a[4]:0,samples);
        }
        vmaDestroyBuffer(g_vk.vma,r->buf,r->mem);
    }bc_result_n=keep;
}
static void bc_finish_frame(void) {
    if(!bc_have_frame)return;
    if(bc_result_n==64){bc_reasons|=BC_CAPACITY;vkq_drain();vk_wait_seq(submit_seq);bc_collect();}
    if(bc_result_n==64)FATAL("A1 result ring did not drain\n");
    BcResult *r=&bc_results[bc_result_n++];memset(r,0,sizeof *r);
    VkBufferCreateInfo bi={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=16+(bc_audit?BC_AUDIT_BYTES:0),.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai={.usage=VMA_MEMORY_USAGE_AUTO_PREFER_HOST,.flags=VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT|VMA_ALLOCATION_CREATE_MAPPED_BIT};VmaAllocationInfo info;
    VK_CHECK(vmaCreateBuffer(g_vk.vma,&bi,&ai,&r->buf,&r->mem,&info));r->ptr=info.pMappedData;
    VkCommandBuffer cb=vk_cmd();bc_before_submit();vk_full_barrier(cb);VkBufferCopy copy={0,0,16};vkCmdCopyBuffer(cb,bc_dirty,r->buf,1,&copy);
    if(bc_audit){VkBufferCopy audit={bc_dirty_size,16,BC_AUDIT_BYTES};vkCmdCopyBuffer(cb,bc_dirty,r->buf,1,&audit);}
    r->seq=vk_pending_seq();r->frame=bc_frame;r->generation=bc_generation;r->on=bc_active;r->reasons=bc_reasons;
    uint64_t done=vk_completed_seq();if(bc_active&&bc_copy_seq>done)r->reasons|=BC_INPUT_COPY;if(bc_active&&bc_vertex_seq>done)r->reasons|=BC_LIVE_VERTEX;
    r->copy_bytes=bc_copy_bytes;r->aliases=bc_aliases;r->dispatches=bc_dispatches;r->draws=bc_draws;r->rb_bytes=bc_rb_bytes;r->rb_skipped=bc_rb_skipped;
    r->updates=bc_updates;r->drains=bc_update_drains;r->cost=bc_cost;memcpy(r->gfx_draws,bc_gfx_draws,sizeof bc_gfx_draws);memcpy(r->gfx_unbounded,bc_gfx_unbounded,sizeof bc_gfx_unbounded);
    vk_submit_extra(VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE);
}
static void bc_begin(void) {
    if(!vk_buffer_cache_configured())return;
    uint64_t frame=renderer_present_count();if(bc_have_frame&&frame==bc_frame)return;
    bc_buffers();bc_finish_frame();bc_collect();
    if(!bc_first_ns)bc_first_ns=plat_now_ns();
    double sec=(plat_now_ns()-bc_first_ns)/1e9;int block=sec<45?-1:(int)((sec-45)/10);
    bool on=bc_ab?(block>=0&&(block%4==1||block%4==2)):bc_mode!=0;
    if(bc_have_frame&&on!=bc_active){vkq_drain();vk_wait_seq(submit_seq);TRACE_LOG("[a1-toggle] ns %llu frame %llu block %d on %u\n",(unsigned long long)plat_now_ns(),(unsigned long long)frame,block,on);}
    bc_active=on;bc_have_frame=true;bc_frame=frame;bc_generation++;bc_map_version++;
    pthread_mutex_lock(&bc_lock);bc_n=0;pthread_mutex_unlock(&bc_lock);
    bc_copy_seq=bc_vertex_seq=bc_copy_bytes=bc_aliases=bc_dispatches=bc_draws=bc_rb_bytes=bc_rb_skipped=0;bc_reasons=0;
    bc_updates=bc_update_drains=0;memset(&bc_cost,0,sizeof bc_cost);memset(bc_gfx_draws,0,sizeof bc_gfx_draws);memset(bc_gfx_unbounded,0,sizeof bc_gfx_unbounded);
    uint32_t zero[4]={0};VkSpan z=vk_transient_copy(zero,16,256);VkBufferCopy c={z.off,0,16};VkCommandBuffer cb=vk_cmd();vk_full_barrier(cb);vkCmdCopyBuffer(cb,z.buf,bc_dirty,1,&c);vk_full_barrier(cb);
    uint32_t empty[4]={0};VkSpan e=vk_transient_copy(empty,16,256);bc_map=(VkDescriptorBufferInfo){e.buf,e.off,16};
    bc_graphics_map[0]=bc_graphics_map[1]=bc_map;
    if(bc_audit){uint32_t zeros[BC_AUDIT_BYTES/4]={0};VkSpan a=vk_transient_copy(zeros,sizeof zeros,256);VkBufferCopy ac={a.off,bc_dirty_size,sizeof zeros};vkCmdCopyBuffer(cb,a.buf,bc_dirty,1,&ac);vk_full_barrier(cb);}
}
// Timestamp only the transfer-stage interval around each snapshot copy.
// No waits are added; results retire with the submission's normal timeline.
#define A1D_COPY_PAIRS 512
static VkQueryPool bc_snapshot_pool;
static struct {uint64_t seq,frame;unsigned n,dropped;} bc_snapshots[SUBS];
static void bc_snapshot_retire(unsigned slot){
    if(!bc_snapshot_pool||!bc_snapshots[slot].n)return;
    unsigned n=bc_snapshots[slot].n;uint64_t ticks[A1D_COPY_PAIRS*2];double ns=0;
    VkResult r=vkGetQueryPoolResults(g_vk.dev,bc_snapshot_pool,slot*A1D_COPY_PAIRS*2,n*2,sizeof ticks,ticks,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
    if(r==VK_SUCCESS)for(unsigned i=0;i<n;i++)ns+=(ticks[2*i+1]-ticks[2*i])*g_vk.props.limits.timestampPeriod;
    TRACE_LOG("[a1d-snapshot] frame %llu seq %llu copies %u gpu_ns %.0f dropped %u result %d\n",
        (unsigned long long)bc_snapshots[slot].frame,(unsigned long long)bc_snapshots[slot].seq,n,ns,bc_snapshots[slot].dropped,r);
    bc_snapshots[slot].n=0;
}
static unsigned bc_snapshot_begin(VkCommandBuffer cb){
    if(!a1d_enabled)return ~0u;
    if(!bc_snapshot_pool){VkQueryPoolCreateInfo ci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=SUBS*A1D_COPY_PAIRS*2};VK_CHECK(vkCreateQueryPool(g_vk.dev,&ci,NULL,&bc_snapshot_pool));}
    typeof(bc_snapshots[0]) *s=&bc_snapshots[cur_sub];
    if(s->seq!=vk_pending_seq()){
        s->seq=vk_pending_seq();s->frame=renderer_present_count();s->n=s->dropped=0;
        vkCmdResetQueryPool(cb,bc_snapshot_pool,cur_sub*A1D_COPY_PAIRS*2,A1D_COPY_PAIRS*2);
    }
    if(s->n==A1D_COPY_PAIRS){s->dropped++;return ~0u;}
    unsigned q=cur_sub*A1D_COPY_PAIRS*2+2*s->n++;
    vkCmdWriteTimestamp2(cb,VK_PIPELINE_STAGE_2_TRANSFER_BIT,bc_snapshot_pool,q);return q;
}
static void bc_snapshot_end(VkCommandBuffer cb,unsigned q){if(q!=~0u)vkCmdWriteTimestamp2(cb,VK_PIPELINE_STAGE_2_TRANSFER_BIT,bc_snapshot_pool,q+1);}
static bool bc_ensure_range(uint64_t addr,uint64_t size) {
    if(!addr||!size)return true;
    if((addr&3)||(size&3)||addr<HEAP_BASE||addr-HEAP_BASE>=bc_size||size>bc_size-(addr-HEAP_BASE)||size>(32u<<20)){bc_reasons|=BC_BAD_RANGE;return false;}
    uint64_t lo=addr-HEAP_BASE,hi=lo+size;
    if(bc_covers_locked(lo,hi))return true;
    if(bc_n==BC_MAX){bc_reasons|=BC_CAPACITY;return false;}
    // Earlier descriptor jobs retain their immutable BcView. Growth may publish
    // a new generation without waiting for those jobs or changing their aliases.
    uint64_t begin=plat_now_ns();
    vk_end_pass();
    VkCommandBuffer cb=vk_cmd();bc_barrier(cb);
    // Copy only holes. Re-copying an overlap would overwrite an earlier GPU
    // producer with stale guest bytes. The upload snapshots CPU bytes now;
    // unsupported original writers are drained before this path is reached.
    uint64_t cursor=lo;
    for(unsigned i=0;i<=bc_n&&cursor<hi;i++){
        uint64_t begin=i<bc_n?bc_ranges[i].lo:hi,end=i<bc_n?bc_ranges[i].hi:hi;
        if(end<=cursor)continue;
        if(begin>cursor){uint64_t limit=begin<hi?begin:hi;VkSpan s=vk_transient_copy((const void *)(uintptr_t)(HEAP_BASE+cursor),limit-cursor,256);VkBufferCopy c={s.off,cursor,limit-cursor};unsigned q=bc_snapshot_begin(cb);vkCmdCopyBuffer(cb,s.buf,bc_buffer,1,&c);bc_snapshot_end(cb,q);bc_copy_bytes+=c.size;bc_cost.snapshot_n++;cursor=limit;}
        if(end>cursor)cursor=end;
    }
    bc_barrier(cb);
    unsigned i=0;while(i<bc_n&&bc_ranges[i].hi<lo)i++;unsigned j=i;
    while(j<bc_n&&bc_ranges[j].lo<=hi){if(bc_ranges[j].lo<lo)lo=bc_ranges[j].lo;if(bc_ranges[j].hi>hi)hi=bc_ranges[j].hi;j++;}
    pthread_mutex_lock(&bc_lock);memmove(&bc_ranges[i+1],&bc_ranges[j],(bc_n-j)*sizeof bc_ranges[0]);bc_n=bc_n-(j-i)+1;bc_ranges[i]=(typeof(bc_ranges[0])){lo,hi};pthread_mutex_unlock(&bc_lock);
    bc_map_version++;bc_cost.snapshot_ns+=plat_now_ns()-begin;
    return true;
}
static bool bc_supported(const ShaderInfo *si) {
    static struct{const ShaderInfo *si;bool ok;}memo[256];static unsigned next;
    for(unsigned i=0;i<256;i++)if(memo[i].si==si)return memo[i].ok;
    char *s=buffer_cache_shader(si->msl);bool ok=s!=NULL;free(s);memo[next++&255]=(typeof(memo[0])){si,ok};return ok;
}
static void bc_prepare(const ShaderInfo *si) {
    if(!vk_buffer_cache_configured()||!si->uses_gmem)return;
    if(!bc_active)return;
    if(bc_original_writer){renderer_flush();bc_original_writer=false;}
    if(!bc_supported(si)){bc_reasons|=BC_UNSUPPORTED;bc_before_submit();return;}
    bc_dispatches++;
    for(int kind=0;kind<2;kind++)for(int i=0;i<16;i++){Range r=kind?st.sbuf[5][i]:st.ubuf[5][i];bc_ensure_range(r.addr,r.size);}
    uint32_t map[4+BC_MAX*2]={bc_n,1,(uint32_t)bc_generation,0};
    for(unsigned i=0;i<bc_n;i++){map[4+2*i]=(uint32_t)(bc_ranges[i].lo/4);map[5+2*i]=(uint32_t)(bc_ranges[i].hi/4);}
    uint64_t bytes=(4+2*bc_n)*4;VkSpan s=vk_transient_copy(map,bytes,256);bc_map=(VkDescriptorBufferInfo){s.buf,s.off,bytes};
}
static unsigned bc_shader_slot(const ShaderInfo *si,int stage) {
    for(unsigned i=0;i<bc_nshaders;i++)if(bc_shaders[i].si==si)return i;
    if(bc_nshaders==BC_SHADER_COUNT){bc_reasons|=BC_CAPACITY;return BC_SHADER_COUNT;}
    unsigned i=bc_nshaders++;char *s=buffer_cache_shader_stage(si->msl,stage);
    bc_shaders[i]=(BcShader){si,vk_translate_id(si),stage,buffer_cache_graphics_proved(si->msl,stage),s!=NULL};free(s);
    LOG("[a1b-class] id %016llx stage %d classification %s rewrite %u proof %s\n",(unsigned long long)bc_shaders[i].id,stage,
        bc_shaders[i].proved?"bounded":"unbounded",bc_shaders[i].rewritten,bc_shaders[i].proved?"storage0_plus_instance64_offsets_0_to_44_and_60":"no_address_derivation_proof");
    char path[160];snprintf(path,sizeof path,"build/codex-perf/a1b-buffer-cache/shaders/%016llx.glsl",(unsigned long long)bc_shaders[i].id);
    FILE *f=fopen(path,"wb");if(f){fwrite(si->msl,1,strlen(si->msl),f);fclose(f);}
    return i;
}
static void bc_prepare_graphics(const ShaderInfo *si,int stage,CmdRec *draw,bool indexed) {
    if(!vk_buffer_cache_configured()||!si->uses_gmem)return;
    unsigned slot=bc_shader_slot(si,stage);if(slot==BC_SHADER_COUNT)return;
    BcShader *info=&bc_shaders[slot];bc_gfx_draws[slot]++;
    uint64_t first=(uint32_t)draw->x[indexed?5:3],count=(uint32_t)draw->x[indexed?6:4];if(!count)count=1;
    Range binding=st.sbuf[stage][0];
    bool bounded=info->proved && first+count<=0x04000000ull && (first+count)*64<=binding.size && binding.addr>=HEAP_BASE &&
        !(binding.addr&3) && binding.addr-HEAP_BASE<bc_size && (first+count)*64<=bc_size-(binding.addr-HEAP_BASE);
    if(!bounded)bc_gfx_unbounded[slot]++;
    if(bc_active) {
        if(!info->rewritten){bc_reasons|=BC_UNSUPPORTED;bounded=false;}
        if(bounded){if(!bc_ensure_range(binding.addr+first*64,count*64))bounded=false;}
        if(!bounded){vk_end_pass();bc_before_submit();bc_reasons|=BC_LIVE_GRAPHICS;}
        if(si->writes_gmem&&!bounded){renderer_flush();pthread_mutex_lock(&bc_lock);bc_n=0;pthread_mutex_unlock(&bc_lock);bc_map_version++;bc_original_writer=true;}
    }
    if(bc_audit && bc_gfx_draws[slot]<=4)
        TRACE_LOG("[a1b-bound] frame %llu on %u id %016llx address %llx size %llu first %llu instances %llu bounded %u\n",
            (unsigned long long)bc_frame,bc_active,(unsigned long long)info->id,(unsigned long long)binding.addr,(unsigned long long)binding.size,
            (unsigned long long)first,(unsigned long long)count,bounded);
    uint32_t mode=!bc_active?0:bounded?1:2;
    static struct {uint64_t version,seq;unsigned slot,mode;VkDescriptorBufferInfo map;}memo[2];
    if(memo[stage].version==bc_map_version&&memo[stage].seq==vk_pending_seq()&&memo[stage].slot==slot&&memo[stage].mode==mode&&memo[stage].map.buffer){bc_graphics_map[stage]=memo[stage].map;return;}
    uint32_t map[4+BC_MAX*2]={bc_n,mode,slot,bc_audit!=0};
    for(unsigned i=0;i<bc_n;i++){map[4+2*i]=(uint32_t)(bc_ranges[i].lo/4);map[5+2*i]=(uint32_t)(bc_ranges[i].hi/4);}
    uint64_t bytes=(4+2*bc_n)*4;VkSpan s=vk_transient_copy(map,bytes,256);bc_graphics_map[stage]=(VkDescriptorBufferInfo){s.buf,s.off,bytes};
    memo[stage]=(typeof(memo[0])){bc_map_version,vk_pending_seq(),slot,mode,bc_graphics_map[stage]};
}
static VkPipeline bc_update_pipe;
static VkPipelineLayout bc_update_layout;
static VkDescriptorSetLayout bc_update_set;
static void bc_make_update(void) {
    if(bc_update_pipe)return;
    const char *src="#version 460\nlayout(local_size_x=64) in;\n"
        "layout(std430,binding=0) buffer Owned {uint w[];} owned_;\n"
        "layout(std430,binding=1) buffer Dirty {uint w[];} dirty_;\n"
        "layout(std430,binding=2) readonly buffer Data {uint w[];} data_;\n"
        "layout(push_constant) uniform P {uint word;uint first;uint bytes;uint src;} p;\n"
        "void main(){uint k=gl_GlobalInvocationID.x;if(k*4u>=p.first+p.bytes)return;"
        "uint i=p.word+k,v=owned_.w[i];for(uint b=0u;b<4u;b++){uint x=k*4u+b;"
        "if(x>=p.first&&x<p.first+p.bytes){uint a=p.src+x-p.first;uint y=(data_.w[a>>2u]>>((a&3u)*8u))&255u;"
        "v=(v&~(255u<<(b*8u)))|(y<<(b*8u));}}owned_.w[i]=v;"
        "uint bit=1u<<(i&31u);uint old=atomicOr(dirty_.w[4u+(i>>5u)],bit);"
        "if((old&bit)==0u)atomicAdd(dirty_.w[1],4u);}";
    size_t nw=0;uint32_t *spv=vk_glsl_to_spirv(src,STAGE_COMPUTE,"a1c-ordered-uniform",&nw);
    if(!spv)FATAL("A1c ordered-uniform shader failed\n");VkShaderModule mod=vk_module(spv,nw);free(spv);
    VkBindingInfo bindings[3]={{0,0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},{0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},{0,2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}};
    bc_update_set=vk_set_layout(bindings,3,0,VK_SHADER_STAGE_COMPUTE_BIT);
    VkPushConstantRange push={VK_SHADER_STAGE_COMPUTE_BIT,0,16};
    VkPipelineLayoutCreateInfo li={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&bc_update_set,.pushConstantRangeCount=1,.pPushConstantRanges=&push};
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev,&li,NULL,&bc_update_layout));
    VkComputePipelineCreateInfo ci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.layout=bc_update_layout,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=mod,.pName="main"}};
    VK_CHECK(vkCreateComputePipelines(g_vk.dev,VK_NULL_HANDLE,1,&ci,NULL,&bc_update_pipe));vkDestroyShaderModule(g_vk.dev,mod,NULL);
}
static void bc_uniform_update(uint64_t addr,uint64_t size,const void *data) {
    if(!bc_active||!size||addr<HEAP_BASE)return;
    uint64_t lo=addr-HEAP_BASE,hi=lo+size;bool overlap=false;
    for(unsigned i=0;i<bc_n;i++)overlap|=lo<bc_ranges[i].hi&&bc_ranges[i].lo<hi;
    if(!overlap)return;
    uint64_t begin=plat_now_ns();bc_updates++;bc_cost.uniform_n++;bc_cost.uniform_inpass+=in_pass;
    vk_end_pass();bc_make_update();VkCommandBuffer cb=vk_cmd();bc_barrier(cb);
    VkSpan upload=vk_transient((size+3)&~3ull,256);memcpy(upload.ptr,data,size);
    if(size&3)memset((uint8_t*)upload.ptr+size,0,4-(size&3));
    VkDescriptorBufferInfo bi[3]={{bc_buffer,0,bc_size},{bc_dirty,0,bc_dirty_size},{upload.buf,upload.off,(size+3)&~3ull}};
    VkDescriptorSet ds=vk_alloc_set(bc_update_set);VkWriteDescriptorSet w[3];
    for(unsigned i=0;i<3;i++)w[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
    vkUpdateDescriptorSets(g_vk.dev,3,w,0,NULL);vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,bc_update_pipe);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,bc_update_layout,0,1,&ds,0,NULL);
    for(unsigned i=0;i<bc_n;i++){
        uint64_t a=lo>bc_ranges[i].lo?lo:bc_ranges[i].lo,b=hi<bc_ranges[i].hi?hi:bc_ranges[i].hi;if(a>=b)continue;
        uint32_t push[4]={(uint32_t)(a/4),(uint32_t)(a&3),(uint32_t)(b-a),(uint32_t)(a-lo)};
        vkCmdPushConstants(cb,bc_update_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,push);
        vkCmdDispatch(cb,(push[1]+push[2]+255)/256,1,1);
        bc_cost.update_gpu_n++;bc_cost.update_gpu_bytes+=b-a;
    }
    bc_barrier(cb);bc_writes=true;bc_cost.uniform_ns+=plat_now_ns()-begin;
}
static void bc_cpu_change(uint64_t addr,uint64_t size,bool uniform) {
    if(!bc_active||!size||addr<HEAP_BASE)return;
    bool overlap=false;for(unsigned i=0;i<bc_n;i++)overlap|=addr-HEAP_BASE<bc_ranges[i].hi&&bc_ranges[i].lo<addr-HEAP_BASE+size;
    if(!overlap)return;
    uint64_t begin=plat_now_ns(),wb=bc_cost.wb_ns;
    if(uniform){bc_cost.uniform_n++;bc_cost.uniform_inpass+=in_pass;}else bc_cost.other_drains++;
    bc_updates++;renderer_flush();bc_update_drains++;
    if(uniform){bc_cost.uniform_ns+=plat_now_ns()-begin;bc_cost.uniform_wb_ns+=bc_cost.wb_ns-wb;}
    // Prior writes are published. Preserve every unaffected interval and remove
    // only complete words overlapping the CPU update, including partial writes.
    uint64_t lo=(addr-HEAP_BASE)&~3ull,hi=(addr-HEAP_BASE+size+3)&~3ull;
    typeof(bc_ranges[0]) kept[BC_MAX+1];unsigned n=0;
    for(unsigned i=0;i<bc_n;i++) {
        uint64_t a=bc_ranges[i].lo,b=bc_ranges[i].hi;
        if(b<=lo||a>=hi)kept[n++]=bc_ranges[i];
        else {if(a<lo)kept[n++]=(typeof(kept[0])){a,lo};if(hi<b)kept[n++]=(typeof(kept[0])){hi,b};}
    }
    pthread_mutex_lock(&bc_lock);
    if(n>BC_MAX){bc_reasons|=BC_CPU_UPDATE|BC_CAPACITY;bc_n=0;}
    else {memcpy(bc_ranges,kept,n*sizeof kept[0]);bc_n=n;}
    pthread_mutex_unlock(&bc_lock);bc_generation++;bc_map_version++;
}
static void bc_reset(void) {
    if(bc_buffer)vmaDestroyBuffer(g_vk.vma,bc_buffer,bc_mem);
    if(bc_dirty)vmaDestroyBuffer(g_vk.vma,bc_dirty,bc_dirty_mem);
    for(unsigned i=0;i<bc_result_n;i++)vmaDestroyBuffer(g_vk.vma,bc_results[i].buf,bc_results[i].mem);
    bc_buffer=bc_dirty=VK_NULL_HANDLE;bc_n=bc_result_n=bc_nshaders=0;bc_have_frame=bc_active=bc_writes=false;bc_size=0;
}
