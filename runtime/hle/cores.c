// The game's core count (BOTW-JOBS-NATIVE.md). sead hard-codes 3 (sead::CoreInfo::configure); the runtime's native
// per-core structures (runtime/hle/vfr.c, ...) are sized for up to 32 and bound their slots by this count. It stays 3
// until every per-core structure the game's jobs use has its N slots: then the core count patch gives the game the
// host's cores and this returns the same number.
int rt_game_cores(void) { return 3; }
