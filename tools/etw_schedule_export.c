// Export Windows kernel scheduling records with raw QPC ticks. Read-only,
// offline ETL consumer; TDH supplies field layouts rather than guessed offsets.
// Build: gcc -std=c11 -O2 -Wall -Wextra -Werror tools/etw_schedule_export.c
//        -o build/codex-perf/etw-schedule-export.exe -ltdh -ladvapi32
// Microsoft CSwitch/ReadyThread schemas:
// https://learn.microsoft.com/en-us/windows/win32/etw/cswitch
// https://learn.microsoft.com/en-us/windows/win32/etw/readythread
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <evntrace.h>
#include <evntcons.h>
#include <tdh.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const GUID thread_provider = {0x3d6fa8d1,0xfe05,0x11d0,{0x9d,0xda,0x00,0xc0,0x4f,0xd7,0xba,0x7c}};
static const GUID sample_provider = {0xce1dbfb4,0x137e,0x4da6,{0x87,0xb0,0x3f,0x59,0xaa,0x10,0x2c,0xbc}};
static const GUID image_provider = {0x2cb15d1d,0x5fc1,0x11d2,{0xab,0xe1,0x00,0xa0,0xc9,0x11,0xf5,0x18}};
static const GUID stack_provider = {0xdef2fe46,0x7bd6,0x4b80,{0xbd,0x94,0xf5,0x7f,0xe2,0x0d,0x0c,0xe3}};
static FILE *output;
static FILE *names_output;
static FILE *modules_output;
static FILE *stacks_output;
static uint32_t stack_pid;
static uint64_t stacks, stack_other[256];
static uint64_t samples, images;
static uint64_t switches, ready, threads, failures;
static ULONG lost;
static unsigned char seen_opcode[256];

static int property(EVENT_RECORD *event, const wchar_t *name, uint64_t *value) {
    PROPERTY_DATA_DESCRIPTOR descriptor = {0};
    descriptor.PropertyName = (ULONGLONG)(uintptr_t)name;
    descriptor.ArrayIndex = ULONG_MAX;
    *value = 0;
    return TdhGetProperty(event, 0, NULL, 1, &descriptor, sizeof(*value), (BYTE *)value) == ERROR_SUCCESS;
}

static void WINAPI event_record(EVENT_RECORD *event) {
    unsigned event_opcode=event->EventHeader.EventDescriptor.Opcode;
    uint64_t event_tick=(uint64_t)event->EventHeader.TimeStamp.QuadPart;
    if (stacks_output && !memcmp(&event->EventHeader.ProviderId,&stack_provider,sizeof(GUID))) {
        if (event_opcode != 32) { stack_other[event_opcode]++; return; }
        // StackWalkStackTraceData in Microsoft's TraceEvent: original event QPC
        // at0, PID at8, TID at12, then pointer-sized PCs (leaf first). The header
        // timestamp is NOT the timestamp of the event this stack belongs to.
        unsigned pointer_size = event->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER ? 4 : 8;
        unsigned bytes=event->UserDataLength;
        if (bytes < 16 || (bytes-16)%pointer_size) { failures++; return; }
        const BYTE *data=event->UserData;
        uint64_t original_tick; uint32_t pid,tid;
        memcpy(&original_tick,data,8); memcpy(&pid,data+8,4); memcpy(&tid,data+12,4);
        if (pid != stack_pid) return;
        fprintf(stacks_output,"%" PRIu64 "\t%u\t%u",original_tick,pid,tid);
        for (unsigned at=16;at<bytes;at+=pointer_size) {
            uint64_t pc=0; memcpy(&pc,data+at,pointer_size);
            fprintf(stacks_output,"\t%" PRIx64,pc);
        }
        fputc('\n',stacks_output); stacks++; return;
    }
    if (!memcmp(&event->EventHeader.ProviderId,&sample_provider,sizeof(GUID)) && event_opcode==46) {
        uint64_t tid,ip;
        if ((!property(event,L"ThreadId",&tid) && !property(event,L"TThreadId",&tid)) ||
            !property(event,L"InstructionPointer",&ip)) { failures++; return; }
        fprintf(output,"%" PRIu64 ",S,%u,%" PRIu64 ",%" PRIu64 ",0,0,0,0\n",event_tick,event->BufferContext.ProcessorNumber,tid,ip);
        samples++; return;
    }
    if (!memcmp(&event->EventHeader.ProviderId,&image_provider,sizeof(GUID)) &&
        (event_opcode==10 || event_opcode==3 || event_opcode==4)) {
        uint64_t pid,base,size;
        if (!property(event,L"ProcessId",&pid) || !property(event,L"ImageBase",&base) ||
            !property(event,L"ImageSize",&size)) { failures++; return; }
        PROPERTY_DATA_DESCRIPTOR descriptor={0};
        descriptor.PropertyName=(ULONGLONG)(uintptr_t)L"FileName"; descriptor.ArrayIndex=ULONG_MAX;
        wchar_t name[2048]={0}; char utf8[8192]={0};
        if (TdhGetProperty(event,0,NULL,1,&descriptor,sizeof(name),(BYTE *)name)==ERROR_SUCCESS) {
            name[2047]=0; WideCharToMultiByte(CP_UTF8,0,name,-1,utf8,sizeof(utf8),NULL,NULL);
            for (char *s=utf8;*s;s++) if (*s=='\t'||*s=='\r'||*s=='\n') *s=' ';
        }
        fprintf(modules_output,"%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%s\n",pid,base,size,utf8);
        images++; return;
    }
    if (memcmp(&event->EventHeader.ProviderId, &thread_provider, sizeof(GUID))) return;
    unsigned opcode = event->EventHeader.EventDescriptor.Opcode;
    if (!seen_opcode[opcode]) {
        seen_opcode[opcode] = 1;
        ULONG size = 0;
        ULONG info_status = TdhGetEventInformation(event,0,NULL,NULL,&size);
        if (info_status != ERROR_INSUFFICIENT_BUFFER) {
            fprintf(stderr,"Thread opcode%u version%u bytes%u TDH status%lu\n",opcode,event->EventHeader.EventDescriptor.Version,event->UserDataLength,info_status);
        }
        if (info_status == ERROR_INSUFFICIENT_BUFFER) {
            TRACE_EVENT_INFO *info = malloc(size);
            if (info && TdhGetEventInformation(event,0,NULL,info,&size) == ERROR_SUCCESS) {
                fprintf(stderr,"Thread opcode%u version%u properties:",opcode,event->EventHeader.EventDescriptor.Version);
                for (ULONG i=0;i<info->TopLevelPropertyCount;i++) {
                    const wchar_t *name = (const wchar_t *)((BYTE *)info + info->EventPropertyInfoArray[i].NameOffset);
                    fprintf(stderr," %ls",name);
                }
                fputc('\n',stderr);
            }
            free(info);
        }
    }
    uint64_t tick = (uint64_t)event->EventHeader.TimeStamp.QuadPart;
    unsigned cpu = event->BufferContext.ProcessorNumber;
    uint64_t a, b, state = 0, reason = 0, priority = 0, pid = 0;
    char kind;
    if (opcode == 36) {
        // Windows26100 emits v5, absent from its installed TDH schema. Microsoft's
        // TraceEvent decoder retains these offsets for all versions>2 with>=24B.
        // Restrict this fallback to the observed v5/28B shape, failing closed.
        // github.com/microsoft/perfview/blob/main/src/TraceEvent/Parsers/KernelTraceEventParser.cs
        if (event->EventHeader.EventDescriptor.Version == 5 && event->UserDataLength == 28) {
            const BYTE *data = event->UserData;
            uint32_t new_id, old_id;
            memcpy(&new_id,data,4); memcpy(&old_id,data+4,4);
            a=new_id; b=old_id; priority=data[8]; reason=data[12]; state=data[14];
        } else if (!property(event,L"NewThreadId",&a) || !property(event,L"OldThreadId",&b) ||
            !property(event,L"OldThreadState",&state) || !property(event,L"OldThreadWaitReason",&reason) ||
            !property(event,L"NewThreadPriority",&priority)) { failures++; return; }
        // Preserve the raw state/priority, including newer Windows state values;
        // the analyzer only treats explicitly known ready states as runnable.
        kind = 'C'; switches++;
    } else if (opcode == 50) {
        if (!property(event,L"TThreadId",&a)) { failures++; return; }
        b = event->EventHeader.ThreadId;
        pid = event->EventHeader.ProcessId;
        kind = 'R'; ready++;
    } else if (opcode >= 1 && opcode <= 4) {
        if (!property(event,L"ProcessId",&pid) ||
            (!property(event,L"TThreadId",&a) && !property(event,L"ThreadId",&a))) { failures++; return; }
        b = 0;
        kind = opcode == 2 || opcode == 4 ? 'E' : 'T';
        // DC end (opcode4) is a rundown, not a thread termination. Keep it T.
        if (opcode == 4) kind = 'T';
        threads++;
    } else if (opcode == 72) {
        if (!property(event,L"ProcessId",&pid) || !property(event,L"ThreadId",&a)) { failures++; return; }
        PROPERTY_DATA_DESCRIPTOR descriptor = {0};
        descriptor.PropertyName = (ULONGLONG)(uintptr_t)L"ThreadName";
        descriptor.ArrayIndex = ULONG_MAX;
        wchar_t name[2048] = {0};
        if (TdhGetProperty(event,0,NULL,1,&descriptor,sizeof(name),(BYTE *)name) == ERROR_SUCCESS) {
            char utf8[8192]; name[2047]=0;
            if (WideCharToMultiByte(CP_UTF8,0,name,-1,utf8,sizeof(utf8),NULL,NULL)) {
                for (char *s=utf8;*s;s++) if (*s=='\t'||*s=='\r'||*s=='\n') *s=' ';
                fprintf(names_output,"%" PRIu64 "\t%" PRIu64 "\t%s\n",pid,a,utf8);
            }
        }
        return;
    } else return;
    fprintf(output,"%" PRIu64 ",%c,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
            tick,kind,cpu,a,b,state,reason,pid,priority);
}

static ULONG WINAPI buffer_record(EVENT_TRACE_LOGFILEW *log) {
    if (log->EventsLost > lost) lost = log->EventsLost;
    return TRUE;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) { fprintf(stderr,"Usage: etw-schedule-export input.etl output.csv [stack-pid]\n"); return 2; }
    if (argc == 4) {
        char *end; unsigned long value=strtoul(argv[3],&end,10);
        if (!value || *end) { fprintf(stderr,"Invalid stack PID\n"); return 2; }
        stack_pid=(uint32_t)value;
    }
    wchar_t path[32768];
    if (!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,argv[1],-1,path,32768)) return 2;
    EVENT_TRACE_LOGFILEW log = {0};
    log.LogFileName = path;
    log.ProcessTraceMode = PROCESS_TRACE_MODE_EVENT_RECORD | PROCESS_TRACE_MODE_RAW_TIMESTAMP;
    log.EventRecordCallback = event_record;
    log.BufferCallback = buffer_record;
    TRACEHANDLE handle = OpenTraceW(&log);
    if (handle == INVALID_PROCESSTRACE_HANDLE) { fprintf(stderr,"OpenTrace failed: %lu\n",GetLastError()); return 1; }
    unsigned clock = log.LogfileHeader.ReservedFlags;
    if (clock != 1 || log.LogfileHeader.PerfFreq.QuadPart <= 0) {
        fprintf(stderr,"Unsupported trace clock %u; refusing to treat it as QPC.\n",clock);
        CloseTrace(handle); return 1;
    }
    output = fopen(argv[2],"wb");
    if (!output) { perror("output"); CloseTrace(handle); return 1; }
    char *names_path = malloc(strlen(argv[2])+20);
    if (!names_path) { fclose(output); CloseTrace(handle); return 1; }
    sprintf(names_path,"%s.names.tsv",argv[2]);
    names_output=fopen(names_path,"wb");
    sprintf(names_path,"%s.modules.tsv",argv[2]);
    modules_output=fopen(names_path,"wb");
    if (stack_pid) {
        sprintf(names_path,"%s.stacks.tsv",argv[2]);
        stacks_output=fopen(names_path,"wb");
    }
    free(names_path);
    if (!names_output) { fclose(output); CloseTrace(handle); return 1; }
    if (!modules_output) { fclose(output); fclose(names_output); CloseTrace(handle); return 1; }
    if (stack_pid && !stacks_output) { fclose(output); fclose(names_output); fclose(modules_output); CloseTrace(handle); return 1; }
    if (stacks_output) setvbuf(stacks_output,NULL,_IOFBF,1<<20);
    setvbuf(output,NULL,_IOFBF,1<<20);
    fprintf(output,"# qpc_frequency=%lld\n",(long long)log.LogfileHeader.PerfFreq.QuadPart);
    fputs("tick,kind,cpu,tid,other_tid,old_state,old_reason,pid,new_priority\n",output);
    ULONG status = ProcessTrace(&handle,1,NULL,NULL);
    CloseTrace(handle);
    int write_error = ferror(output);
    if (fclose(output)) write_error = 1;
    if (fclose(names_output)) write_error = 1;
    if (fclose(modules_output)) write_error = 1;
    if (stacks_output && fclose(stacks_output)) write_error = 1;
    fprintf(stderr,"switches=%" PRIu64 " ready=%" PRIu64 " thread_records=%" PRIu64 " field_failures=%" PRIu64 " events_lost=%lu process_status=%lu\n",
            switches,ready,threads,failures,lost,status);
    fprintf(stderr,"samples=%" PRIu64 " images=%" PRIu64 "\n",samples,images);
    if (stack_pid) {
        fprintf(stderr,"stacks=%" PRIu64 " filtered_pid=%u\n",stacks,stack_pid);
        for (unsigned op=0;op<256;op++) if (stack_other[op])
            fprintf(stderr,"Unexported stack opcode%u count=%" PRIu64 "\n",op,stack_other[op]);
    }
    return status != ERROR_SUCCESS || failures || lost || write_error || !switches || !ready;
}
