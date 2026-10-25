// Real guard-page checks: compact records must never write into the next record
// or read arguments beyond their stored prefix. Also walk a dense mixed stream.
#include <windows.h>
#include <assert.h>
#include <stdio.h>
#include "../runtime/nvn/command_record.h"

int main(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    size_t page = si.dwPageSize;
    uint8_t *memory = VirtualAlloc(NULL, page * 3, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    assert(memory);
    DWORD previous;
    assert(VirtualProtect(memory + page, page, PAGE_READWRITE, &previous));
    uint64_t args[7];
    uint32_t float_bits[4] = {0x80000000, 0x7fc12345, 0x00000001, 0x7f7fffff};
    float floats[4];
    memcpy(floats, float_bits, sizeof floats);
    for (unsigned i = 0; i < 7; i++) args[i] = 0x1234567800000000ull + i;
    unsigned checked = 0;
    for (unsigned op = 0; op < CMD_COUNT; op++) for (unsigned compact = 0; compact < 2; compact++) {
        unsigned header = compact ? nvn_cmd_compact_header(op) : sizeof(CmdRec);
        assert(header >= 8 && header <= sizeof(CmdRec) && !(header & 7));
        for (unsigned payload = 0; payload < 64; payload++) {
            unsigned size = (header + payload + 7) & ~7u;
            uint8_t *at = memory + 2 * page - size;
            memset(at - 16, 0xa5, 16);
            CmdRec *r = (CmdRec *)at;
            nvn_cmd_init(r, op, size, header);
            nvn_cmd_arguments(r, args, floats);
            for (unsigned i = 0; i < payload; i++) nvn_cmd_extra(r)[i] = (uint8_t)(op + i);
            assert(r->op == op && r->size == size && r->header_size == header);
            assert(nvn_cmd_extra(r) == at + header);
            for (unsigned i = 0; i < 16; i++) assert(at[-16 + (int)i] == 0xa5);
            unsigned kept = header < 64 ? (header - 8) / 8 : 7;
            for (unsigned i = 0; i < 7; i++)
                assert(nvn_cmd_debug_arg(r, i) == (i < kept ? args[i] : 0));
            if (header == sizeof *r) assert(!memcmp(r->f, float_bits, sizeof float_bits));
            for (unsigned i = 0; i < payload; i++) assert(nvn_cmd_extra(r)[i] == (uint8_t)(op + i));
            checked++;
        }
    }
    _Alignas(8) uint8_t stream[CMD_COUNT * 128];
    unsigned used = 0;
    for (unsigned op = 0; op < CMD_COUNT; op++) {
        unsigned h = nvn_cmd_compact_header(op), size = h + 16;
        CmdRec *r = (CmdRec *)(stream + used);
        nvn_cmd_init(r, op, size, h);
        nvn_cmd_arguments(r, args, floats);
        memset(nvn_cmd_extra(r), (int)op, 16);
        used += size;
    }
    unsigned offset = 0;
    for (unsigned op = 0; op < CMD_COUNT; op++) {
        CmdRec *r = (CmdRec *)(stream + offset);
        assert(r->op == op);
        for (unsigned i = 0; i < 16; i++) assert(nvn_cmd_extra(r)[i] == op);
        offset += r->size;
    }
    assert(offset == used);
    VirtualFree(memory, 0, MEM_RELEASE);
    printf("%u guarded record/payload cases and dense mixed-command traversal passed\n", checked);
    return 0;
}
