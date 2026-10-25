// Deterministic clock, actual recorder/writer: exercise a 1.1 ms microspike,
// capped-frame rejection, surrounding-frame retention and JSON escaping.
#include <assert.h>
#include <time.h>
#include "../runtime/nvn/vulkan/frame_trace.c"

static uint64_t test_now = 1000000000ull;
uint64_t plat_now_ns(void) { return test_now; }
static void env(const char *name, const char *value) {
#ifdef _WIN32
    assert(!_putenv_s(name, value));
#else
    assert(!setenv(name, value, 1));
#endif
}
int main(int argc, char **argv) {
    assert(argc == 2);
    env("SAVAGEWIND_FRAME_TRACE", argv[1]);
    env("SAVAGEWIND_FRAME_TRACE_START_S", "0");
    env("SAVAGEWIND_FRAME_TRACE_COUNT", "1");
    env("SAVAGEWIND_FRAME_TRACE_LARGE_COUNT", "2");
    vk_frame_trace_init();
    assert(vk_frame_trace_on);
    for (unsigned i = 0; i < 100; i++) {
        test_now += 33000000;
        vk_frame_trace_present(i, 30);
    }
    assert(history_n == 0 && !trigger_frame && captures == 0);
    vk_frame_trace_reset();
    for (unsigned i = 0; i <= 96; i++) {
        test_now += i == 80 ? 6100000 : 5000000;
        vk_frame_trace_event("test \"pass\"\n", 10, i, i * 10, i * 1000000, i * 1000000 + 120000, 7, 0x1234);
        vk_frame_trace_phase("test CPU phase", i, i * 10, test_now - 1000000, test_now, 3, 123456, 42);
        vk_frame_trace_present(i, 0);
        if (i == 80) assert(trigger_frame == 80);
    }
    assert(captures == 1);
    struct timespec pause = {.tv_nsec = 1000000};
    for (int i = 0; i < 5000 && atomic_load(&writing); i++) nanosleep(&pause, NULL);
    assert(!atomic_load(&writing));
    // A second microspike cannot consume the reserved large-hitch budget.
    for (unsigned i = 97; i <= 236; i++) {
        test_now += i == 180 ? 6500000 : i == 220 ? 20000000 : 5000000;
        vk_frame_trace_present(i, 0);
        if (i == 180) assert(!trigger_frame && captures == 1);
        if (i == 220) assert(trigger_frame == 220 && trigger_large);
    }
    assert(captures == 2 && micro_captures == 1 && large_captures == 1);
    for (int i = 0; i < 5000 && atomic_load(&writing); i++) nanosleep(&pause, NULL);
    assert(!atomic_load(&writing));
    // Upgrade a pending micro neighborhood when a larger hitch follows it.
    vk_frame_trace_reset();
    micro_captures = 0;
    for (unsigned i = 0; i <= 106; i++) {
        test_now += i == 80 ? 6100000 : i == 84 ? 21000000 : i == 87 ? 25000000 : i == 89 ? 22000000 : 5000000;
        vk_frame_trace_present(i, 0);
        if (i == 80) assert(trigger_frame == 80 && !trigger_large);
        if (i == 84) assert(trigger_frame == 84 && trigger_large);
        if (i == 87) assert(trigger_frame == 87 && trigger_ms == 25 && trigger_large);
        if (i == 89) assert(trigger_frame == 87 && trigger_ms == 25); // smaller later hitch must not displace it
    }
    assert(captures == 3 && micro_captures == 0 && large_captures == 2);
    for (int i = 0; i < 5000 && atomic_load(&writing); i++) nanosleep(&pause, NULL);
    assert(!atomic_load(&writing));
    puts("frame trace: cap rejection, micro capture, reserved large budget and worst-hitch upgrade passed");
    return 0;
}
