# One isolated, reproducible run. Saves go into a fresh run directory. No stack
# sampler, screenshot, or per-draw timing in throughput mode. Never stop another
# game instance. The caller should run A/B/B/A without compiling between runs.
param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$GameFiles,
    [Parameter(Mandatory=$true)][string]$SaveTemplate,
    [string]$Exe = "build/savagewind.exe",
    [string]$GameLibrary = "build/swgame.dll",
    [ValidateRange(10,2100)][int]$Seconds = 72,
    [ValidateSet('Normal','AboveNormal')][string]$ProcessPriority = 'Normal',
    [switch]$Diagnostic,
    [ValidateSet("", "main", "jobs", "producer", "descriptor", "recorder", "gpu", "stub", "cpu-fence", "overlap-sensitivity", "overlap-jobs-stub", "overlap-queue-ahead", "overlap-downstream", "job-window", "fence-flags", "safe-overlap")][string]$Survey = "",
    [switch]$GpuTrace,
    [string]$NsightExe = '',
    [ValidateRange(1000,240000)][int]$GpuTraceStartMs = 45000,
    [ValidateRange(1,1000)][int]$GpuTraceDurationMs = 100,
    [ValidateRange(1,100)][int]$GpuTraceFrames = 8,
    [switch]$GpuTraceShaderProfiler,
    [switch]$Validation,
    [ValidateRange(1,100000)][int]$ValidationShotAt = 2100,
    [ValidateRange(0,100000)][int]$ScreenshotAt = 0,
    [switch]$DirectWake,
    [switch]$ScopedBarriers,
    [switch]$GeometryGate,
    [switch]$FrameTrace,
    [switch]$ListStats,
    [switch]$NoShaderWarm,
    [switch]$PipelineLto,
    [switch]$PipelineStats,
    [switch]$MonolithicPipelines,
    [switch]$NativePackets,
    [ValidateRange(0,2)][int]$NativeMultiDraw = 0,
    [ValidateRange(1,128)][int]$NativeBatchLimit = 128,
    [ValidateRange(0,1000000)][int]$NativeMultiDrawABAt = 0,
    [switch]$EarlyFence,
    [switch]$SafeOverlap,
    [switch]$VerifyDependencies,
    [switch]$ReadbackDump,
    [ValidateRange(0,3)][int]$ConsumerProbe = 0,
    [switch]$ConsumerVerify,
    [switch]$ResultDeliveryCoverage,
    [switch]$BufferCache,
    [switch]$BufferCacheAB,
    [switch]$BufferCacheAudit,
    [switch]$A1Attribution,
    [ValidateRange(0,7)][int]$B2 = 0,
    [switch]$B2AB,
    [ValidatePattern('^$|^\d+(,\d+)*$')][string]$ExtraShots = '',
    [ValidateRange(0,10000)][int]$ClockProbeMs = 0,
    [switch]$AddressWait,
    [switch]$ShaderConstants,
    [switch]$SyncProbe,
    [switch]$SvcProfile,
    [switch]$DrawProfile,
    [switch]$DrawBatchProbe,
    [switch]$FeedbackProbe,
    [switch]$SynchronousTraceLog,
    [switch]$BoundedConstants,
    [switch]$BoundedConstantsBindFull,
    [switch]$UniformCopyAudit,
    [ValidateRange(1,1000000)][int]$UniformCopyAuditStride = 1021,
    [ValidateRange(0,2)][int]$ComputeCache = 0,
    [ValidateRange(0,4)][int]$ComputeOptimize = 0,
    [ValidateRange(0,65535)][int]$ComputeTestLimit = 0,
    [ValidateRange(0,2)][int]$RangeQueries = 0,
    [ValidateRange(0,2)][int]$HostUploads = 0,
    [switch]$BatchStats,
    [ValidateRange(0,2)][int]$TextureBindings = 0,
    [ValidateRange(0,2)][int]$StageReuse = 0,
    [ValidateRange(0,2)][int]$ProducerMaterial = 0,
    [ValidateRange(0,1000000)][int]$ProducerMaterialABFrames = 0,
    [ValidateRange(0,1000000)][int]$ProducerMaterialABStart = 5000,
    [ValidateRange(0,2)][int]$TextureHashLanes = 1,
    [switch]$SparseWriteWatch,
    [switch]$WriteWatchProbe,
    [switch]$GeometryScanProbe,
    [switch]$GeometryFlushProbe,
    [ValidateRange(0,2)][int]$GeometryCopyAll = 0,
    [ValidateRange(0,2)][int]$GeometryGuard = 0,
    [switch]$ParallelWatchProbe,
    [ValidateRange(1,4)][int]$WriteWatchWorkers = 1,
    [switch]$TextureHolderAge,
    [switch]$TextureLifetimeProbe,
    [ValidateRange(0,2)][int]$CompactRecords = 0,
    [switch]$DepthMirror,
    [switch]$VerifyDepthMirror,
    [ValidateRange(0,16384)][int]$WriteWatchGapWords = 0,
    [ValidateRange(0,8)][int]$DescriptorWorkers = 1,
    [switch]$CubeLayers,
    [switch]$CubeLayersAB,
    [int]$TraceAt = 0,
    [ValidateRange(0,86400)][double]$TraceStart = 45,
    [ValidateRange(1,100)][int]$TraceCount = 12,
    [ValidateRange(0.05,1000)][double]$TraceDeltaMs = 0.5,
    [ValidateRange(0,100)][double]$TraceRatio = 0.20,
    [ValidateRange(1,1000)][double]$TraceLargeMs = 16.667,
    [ValidateRange(1,100)][double]$TraceLargeRatio = 2,
    [ValidateRange(0,100)][int]$TraceLargeCount = 4,
    [int]$ProbeSmallPass = 0,
    [int]$VerifyCubeLayers = 0,
    [ValidatePattern('^$|^\d+(\.\d+)?:\d+(\.\d+)?(:\d+)?$')][string]$Sample = '',
    [switch]$SampleHostOnly,
    [ValidatePattern('^$|^(0x[0-9a-fA-F]+|[0-9]+)(,(0x[0-9a-fA-F]+|[0-9]+)){0,15}$')][string]$SampleHandles = '',
    [ValidateRange(16,512)][int]$SampleBufferMB = 96,
    [ValidatePattern('^$|^\d+:[A-Z_0-9]+:\d+(,\d+:[A-Z_0-9]+:\d+)*$')][string]$InputAfterLoad = '',
    [int]$RebootAt = 0,
    [switch]$LegacyGate,
    [switch]$FullBarriers
)
$ErrorActionPreference = 'Stop'
if (($B2 -or $B2AB) -and ($BufferCache -or $BufferCacheAB -or $A1Attribution -or $EarlyFence -or $SafeOverlap -or $Survey)) { throw 'B2 requires buffer cache, attribution and overlap experiments OFF.' }
if ($GpuTrace -and (-not $Diagnostic -or -not $NsightExe)) { throw 'GpuTrace requires Diagnostic and an explicit NsightExe path.' }
if (($BufferCache -or $BufferCacheAB) -and (-not $Diagnostic -or $EarlyFence -or $SafeOverlap -or $Survey -or $ComputeCache -or $ComputeOptimize)) { throw 'A1 buffer cache requires Diagnostic, normal fences, and no other GPU experiment.' }
if ($BufferCacheAudit -and -not ($BufferCache -or $BufferCacheAB)) { throw 'Address audit requires A1 cache diagnostic.' }
if ($ConsumerVerify -and $ConsumerProbe -ne 3) { throw 'Consumer verification requires the bounded page sampler (mode3).' }
if ($GpuTrace -and $ProcessPriority -ne 'Normal') { throw 'Measure scheduling priority separately from GPU hardware profiling.' }
if ($GpuTrace -and ($Validation -or $Sample -or $DrawProfile -or $UniformCopyAudit)) { throw 'Collect GPU hardware traces separately from intrusive CPU/validation probes.' }
if ($GpuTraceShaderProfiler -and -not $GpuTrace) { throw 'GpuTraceShaderProfiler requires GpuTrace.' }
if ($PipelineStats -and -not $Diagnostic) { throw 'PipelineStats is diagnostic only.' }
if ($DrawBatchProbe -and -not $Diagnostic) { throw 'DrawBatchProbe is diagnostic only.' }
if ($NativeMultiDrawABAt -and (-not $Diagnostic -or -not $NativeMultiDraw)) { throw 'NativeMultiDrawABAt requires Diagnostic and NativeMultiDraw.' }
if ($StageReuse -eq 2 -and -not $Diagnostic) { throw 'StageReuse 2 is diagnostic only.' }
if ($ProducerMaterial -eq 2 -and -not $Diagnostic) { throw 'ProducerMaterial 2 is diagnostic only.' }
if ($ProducerMaterialABFrames -and $ProducerMaterial -ne 1) { throw 'Producer material alternation requires mode1.' }
if ($ProducerMaterial -and ($StageReuse -or $NativeMultiDraw -or $DrawBatchProbe)) { throw 'Measure producer preparation separately from other binding/batching experiments.' }
if ($TextureHashLanes -eq 2 -and -not $Diagnostic) { throw 'TextureHashLanes 2 is diagnostic only.' }
if ($PipelineLto -and $MonolithicPipelines) { throw 'Choose optimized library linking or monolithic pipelines, not both.' }
if ($WriteWatchProbe -and -not $Diagnostic) { throw 'WriteWatchProbe is diagnostic only.' }
if ($FeedbackProbe -and -not $Diagnostic) { throw 'FeedbackProbe is diagnostic only.' }
if ($UniformCopyAudit -and -not $Diagnostic) { throw 'UniformCopyAudit is diagnostic only.' }
if ($ScreenshotAt -and -not $Diagnostic) { throw 'ScreenshotAt is diagnostic only.' }
if ($BoundedConstantsBindFull -and -not $BoundedConstants) { throw 'BoundedConstantsBindFull requires BoundedConstants.' }
if ($GeometryScanProbe -and -not $Diagnostic) { throw 'GeometryScanProbe is diagnostic only.' }
if ($GeometryFlushProbe -and -not $Diagnostic) { throw 'GeometryFlushProbe is diagnostic only.' }
if ($GeometryGuard -eq 2 -and -not $Diagnostic) { throw 'GeometryGuard 2 is diagnostic only.' }
if (($ComputeOptimize -eq 2 -or $ComputeOptimize -eq 4) -and -not $Diagnostic) { throw 'ComputeOptimize 2/4 are diagnostic only.' }
if ($ComputeOptimize -and $ComputeCache) { throw 'Measure compute optimization separately from the heap-cache experiment.' }
if ($ComputeTestLimit -and (-not $Diagnostic -or $ComputeOptimize -ne 3)) { throw 'ComputeTestLimit requires Diagnostic and ComputeOptimize 3.' }
if ($RangeQueries -eq 2 -and -not $Diagnostic) { throw 'RangeQueries 2 is diagnostic only.' }
if ($InputAfterLoad -and -not $Diagnostic) { throw 'InputAfterLoad is for diagnostic movement captures.' }
if ($ParallelWatchProbe -and -not $Diagnostic) { throw 'ParallelWatchProbe is diagnostic only.' }
if ($TextureLifetimeProbe -and -not $Diagnostic) { throw 'TextureLifetimeProbe is diagnostic only.' }
if ($CompactRecords -eq 2 -and -not $Diagnostic) { throw 'CompactRecords 2 is diagnostic only.' }
if ($VerifyDepthMirror -and -not $Diagnostic) { throw 'VerifyDepthMirror is diagnostic only.' }
if ($Name -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Use letters, digits, dash or underscore for Name.' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (Test-Path -LiteralPath (Join-Path $repo 'build/codex-perf/stop-benchmark-queue')) {
    throw 'Benchmark queue canceled before starting another game instance.'
}
$exePath = (Resolve-Path (Join-Path $repo $Exe)).Path
# swgame.dll imports the host as savagewind.exe; keep that basename even when
# isolating a candidate in a different directory. Ownership checks use it too.
if ([IO.Path]::GetFileName($exePath) -ine 'savagewind.exe') { throw 'An isolated Exe must keep the savagewind.exe basename for the game DLL imports.' }
$gamePath = (Resolve-Path $GameFiles).Path
$savePath = (Resolve-Path $SaveTemplate).Path
if (Get-Process savagewind -ErrorAction SilentlyContinue) { throw 'Another game instance is running.' }
$run = Join-Path $repo "build/codex-perf/runs/$Name"
if (Test-Path -LiteralPath $run) { throw "Run already exists: $run" }
New-Item -ItemType Directory -Path $run | Out-Null
Copy-Item -LiteralPath $savePath -Destination (Join-Path $run 'save') -Recurse
$backup = @{}
Get-ChildItem Env:SAVAGEWIND_* | ForEach-Object { $backup[$_.Name]=$_.Value; Remove-Item -LiteralPath $_.PSPath }
$oldPath = $env:PATH
$vkBackup = @{}
foreach ($key in @('VK_ADD_LAYER_PATH','VK_LAYER_ENABLES','VK_LAYER_VALIDATE_SYNC','VK_VALIDATION_VALIDATE_SYNC','VK_INSTANCE_LAYERS')) {
    $item = Get-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue
    $vkBackup[$key] = if ($item) { $item.Value } else { $null }
    Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue
}
try {
    $env:PATH = 'C:\msys64\ucrt64\bin;' + $oldPath
    $env:SAVAGEWIND_DATA_ROOT = $repo
    if ($BufferCache -or $BufferCacheAB) {
        $a1Cache = Join-Path $repo 'build/codex-perf/a1b-buffer-cache/cache'
        if (-not (Test-Path -LiteralPath $a1Cache)) {
            Copy-Item -LiteralPath (Join-Path $repo 'cache') -Destination $a1Cache -Recurse
        }
        $env:SAVAGEWIND_CACHE_DIR = $a1Cache
        $env:SAVAGEWIND_CACHE_KEEP = '1'
    }
    $env:SAVAGEWIND_SAVE_DIR = Join-Path $run 'save'
    $env:SAVAGEWIND_GAME_LIB = (Resolve-Path (Join-Path $repo $GameLibrary)).Path
    $env:SAVAGEWIND_TEST_WARP = 'plateau'
    $env:SAVAGEWIND_TEST_TIME = '12'
    $env:SAVAGEWIND_FPS = '0'
    $env:SAVAGEWIND_MENU_FPS = '30'
    $env:SAVAGEWIND_HIDDEN = '1'
    $env:SAVAGEWIND_NO_USER_INPUT = '1'
    $env:SAVAGEWIND_AUDIO_GAIN = '0'  # keep audio timing/mixing, mute hidden tests
    $env:SAVAGEWIND_SCRIPT_CLOCK = '1'
    $env:SAVAGEWIND_INPUT = '155:A:4,187:A:4,213:A:4,250:A:4,290:A:4,330:A:4,400:A:4,600:A:4'
    if ($InputAfterLoad) { $env:SAVAGEWIND_INPUT += ',' + $InputAfterLoad }
    $env:SAVAGEWIND_FRAME_LOG = if ($Diagnostic -and -not $Survey -and -not $BufferCacheAB -and -not $B2AB -and -not $B2) { '2' } else { '1' }
    if ($Survey) { $env:SAVAGEWIND_SURVEY = $Survey }
    if ($SynchronousTraceLog) { $env:SAVAGEWIND_SYNC_TRACE_LOG = '1' }
    if ($PipelineStats) { $env:SAVAGEWIND_VK_PIPELINE_STATS = '1' }
    if ($DrawBatchProbe) { $env:SAVAGEWIND_VK_DRAW_BATCH_PROBE = '1' }
    if ($GpuTrace) { $env:SAVAGEWIND_GPU_TRACE = '1' }
    $env:SAVAGEWIND_VK_STAGE_REUSE = "$StageReuse"
    $env:SAVAGEWIND_VK_PRODUCER_MATERIAL = "$ProducerMaterial"
    $env:SAVAGEWIND_VK_PRODUCER_MATERIAL_AB_FRAMES = "$ProducerMaterialABFrames"
    $env:SAVAGEWIND_VK_PRODUCER_MATERIAL_AB_START = "$ProducerMaterialABStart"
    $env:SAVAGEWIND_VK_TEXTURE_HASH_LANES = "$TextureHashLanes"
    if ($LegacyGate) { $env:SAVAGEWIND_VK_LEGACY_GEOMETRY_GATE = '1' }
    if ($FullBarriers) { $env:SAVAGEWIND_VK_FULL_BARRIERS = '1' }
    # Pin both alternatives explicitly even if the runtime's default changes.
    $env:SAVAGEWIND_CORE_DIRECT_WAKE = if ($DirectWake) { '1' } else { '0' }
    if ($ScopedBarriers) { $env:SAVAGEWIND_VK_SCOPED_BARRIERS = '1' }
    if ($GeometryGate) { $env:SAVAGEWIND_VK_GEOMETRY_GATE = '1' }
    if ($FrameTrace) {
        $env:SAVAGEWIND_FRAME_TRACE = Join-Path $run 'frame_trace'
        $env:SAVAGEWIND_FRAME_TRACE_START_S = $TraceStart.ToString([Globalization.CultureInfo]::InvariantCulture)
        $env:SAVAGEWIND_FRAME_TRACE_COUNT = [string]$TraceCount
        $env:SAVAGEWIND_FRAME_TRACE_DELTA_MS = $TraceDeltaMs.ToString([Globalization.CultureInfo]::InvariantCulture)
        $env:SAVAGEWIND_FRAME_TRACE_RATIO = $TraceRatio.ToString([Globalization.CultureInfo]::InvariantCulture)
        $env:SAVAGEWIND_FRAME_TRACE_LARGE_MS = $TraceLargeMs.ToString([Globalization.CultureInfo]::InvariantCulture)
        $env:SAVAGEWIND_FRAME_TRACE_LARGE_RATIO = $TraceLargeRatio.ToString([Globalization.CultureInfo]::InvariantCulture)
        $env:SAVAGEWIND_FRAME_TRACE_LARGE_COUNT = [string]$TraceLargeCount
    }
    if ($ListStats) { $env:SAVAGEWIND_VK_LIST_STATS = '1' }
    if ($NoShaderWarm) { $env:SAVAGEWIND_NO_SHADER_WARM = '1' }
    $env:SAVAGEWIND_VK_GPL_LTO = if ($PipelineLto) { '1' } else { '0' }
    if ($MonolithicPipelines) { $env:SAVAGEWIND_VK_NO_GPL = '1' }
    if ($NativePackets) { $env:SAVAGEWIND_VK_NATIVE_PACKETS = '1' }
    $env:SAVAGEWIND_VK_NATIVE_MULTI_DRAW = [string]$NativeMultiDraw
    $env:SAVAGEWIND_VK_NATIVE_BATCH_LIMIT = [string]$NativeBatchLimit
    if ($NativeMultiDrawABAt) { $env:SAVAGEWIND_VK_NATIVE_MULTI_DRAW_AB = [string]$NativeMultiDrawABAt }
    if ($NativeMultiDraw -and $Diagnostic) { $env:SAVAGEWIND_VK_NATIVE_MULTI_DRAW_STATS = '1' }
    if ($EarlyFence) { $env:SAVAGEWIND_VK_EARLY_FENCE = '1' }
    if ($SafeOverlap) { $env:SAVAGEWIND_VK_SAFE_OVERLAP = '1' }
    if ($ReadbackDump) { $env:SAVAGEWIND_DUMP_LINEAR = '1' }
    if ($ConsumerProbe) { $env:SAVAGEWIND_CONSUMER_PROBE = [string]$ConsumerProbe }
    if ($ConsumerVerify) { $env:SAVAGEWIND_CONSUMER_VERIFY = '1' }
    if ($ResultDeliveryCoverage) { $env:SAVAGEWIND_VK_RESULT_DELIVERY_COVERAGE = '1' }
    if ($BufferCache) { $env:SAVAGEWIND_VK_BUFFER_CACHE = '1' }
    if ($BufferCacheAB) { $env:SAVAGEWIND_VK_BUFFER_CACHE_AB = '1' }
    if ($BufferCacheAudit) { $env:SAVAGEWIND_VK_BUFFER_CACHE_AUDIT = '1' }
    if ($A1Attribution) { $env:SAVAGEWIND_A1D = '1'; $env:SAVAGEWIND_VK_TIMING = '1' }
    if ($B2) { $env:SAVAGEWIND_B2 = [string]$B2 }
    if ($B2AB) { $env:SAVAGEWIND_B2_AB = '1' }
    if ($ClockProbeMs) { $env:SAVAGEWIND_CLOCK_PROBE = [string]$ClockProbeMs }
    if ($VerifyDependencies -or $Validation) { $env:SAVAGEWIND_VK_VERIFY_DEPS = '1' }
    $env:SAVAGEWIND_VK_ADDRESS_WAIT = if ($AddressWait) { '1' } else { '0' }
    $env:SAVAGEWIND_VK_SHADER_CONSTANTS = if ($ShaderConstants) { '1' } else { '0' }
    if ($SyncProbe) { $env:SAVAGEWIND_SYNC_PROBE = '1' }
    if ($SvcProfile) { $env:SAVAGEWIND_PROF_SVC = '1'; $env:SAVAGEWIND_THREAD_DUMP_AT = '46,86,126,146' }
    if ($DrawProfile) { $env:SAVAGEWIND_VK_PROF = '1' }
    if ($FeedbackProbe) { $env:SAVAGEWIND_VK_FEEDBACK_PROBE = Join-Path $run 'feedback' }
    if ($BoundedConstants) { $env:SAVAGEWIND_CBR = '1' }
    if ($BoundedConstantsBindFull) { $env:SAVAGEWIND_CBR_BINDFULL = '1' }
    if ($UniformCopyAudit) {
        $env:SAVAGEWIND_VK_UBO_AUDIT = '1'
        $env:SAVAGEWIND_VK_UBO_AUDIT_STRIDE = [string]$UniformCopyAuditStride
    }
    $env:SAVAGEWIND_VK_COMPUTE_CACHE = [string]$ComputeCache
    $env:SAVAGEWIND_VK_COMPUTE_OPT = [string]$ComputeOptimize
    $env:SAVAGEWIND_VK_COMPUTE_TEST_LIMIT = [string]$ComputeTestLimit
    $env:SAVAGEWIND_VK_RANGE_QUERIES = [string]$RangeQueries
    $env:SAVAGEWIND_VK_HOST_UPLOADS = [string]$HostUploads
    $env:SAVAGEWIND_VK_BATCH_STATS = if ($BatchStats) { '1' } else { '0' }
    $env:SAVAGEWIND_VK_TEXTURE_BINDINGS = [string]$TextureBindings
    $env:SAVAGEWIND_VK_SPARSE_WRITE_WATCH = if ($SparseWriteWatch) { '1' } else { '0' }
    $env:SAVAGEWIND_VK_WW_GAP_WORDS = [string]$WriteWatchGapWords
    if ($WriteWatchProbe) { $env:SAVAGEWIND_VK_WW_PLAN_PROBE = '1' }
    if ($GeometryScanProbe) { $env:SAVAGEWIND_VK_GEOMETRY_SCAN_PROBE = '1' }
    if ($GeometryFlushProbe) { $env:SAVAGEWIND_VK_GEOMETRY_FLUSH_PROBE = '1' }
    $env:SAVAGEWIND_VK_GEOMETRY_COPY_ALL = [string]$GeometryCopyAll
    $env:SAVAGEWIND_VK_GEOMETRY_GUARD = [string]$GeometryGuard
    if ($ParallelWatchProbe) { $env:SAVAGEWIND_VK_WW_PARALLEL_PROBE = '1' }
    $env:SAVAGEWIND_VK_WW_WORKERS = [string]$WriteWatchWorkers
    $env:SAVAGEWIND_VK_TEXTURE_HOLDER_AGE = if ($TextureHolderAge) { '1' } else { '0' }
    if ($TextureLifetimeProbe) { $env:SAVAGEWIND_VK_TEXTURE_LIFETIME_PROBE = '1' }
    $env:SAVAGEWIND_NVN_COMPACT_RECORDS = [string]$CompactRecords
    $env:SAVAGEWIND_VK_DEPTH_MIRROR = if ($VerifyDepthMirror) { '2' } elseif ($DepthMirror) { '1' } else { '0' }
    $env:SAVAGEWIND_VK_MT = [string]$DescriptorWorkers
    $env:SAVAGEWIND_VK_CUBE_LAYERS = if ($CubeLayers) { '1' } else { '0' }
    if ($CubeLayersAB) { $env:SAVAGEWIND_VK_CUBE_LAYERS = '2' }
    if ($TraceAt -gt 0) { $env:SAVAGEWIND_FRAME_TRACE_AT = [string]$TraceAt }
    if ($ProbeSmallPass -gt 0) { $env:SAVAGEWIND_PROBE_SMALL_PASS = [string]$ProbeSmallPass }
    if ($VerifyCubeLayers -gt 0) { $env:SAVAGEWIND_VERIFY_CUBE_LAYERS = [string]$VerifyCubeLayers }
    if ($Sample) {
        $env:SAVAGEWIND_SAMPLE = $Sample
        if ($SampleHostOnly) { $env:SAVAGEWIND_SAMPLE_HOST_ONLY = '1' }
        if ($SampleHandles) { $env:SAVAGEWIND_SAMPLE_HANDLES = $SampleHandles }
        $env:SAVAGEWIND_SAMPLE_BUFFER_MB = [string]$SampleBufferMB
        Copy-Item -LiteralPath $exePath -Destination (Join-Path $run 'savagewind.exe')
        $priorSample = Join-Path $repo 'build/samples.txt'
        if (Test-Path -LiteralPath $priorSample) { Copy-Item -LiteralPath $priorSample -Destination (Join-Path $run 'samples.previous.txt') }
    }
    if ($RebootAt -gt 0) { $env:SAVAGEWIND_REBOOT_AT = "${RebootAt}:game" }
    if ($Validation) {
        $env:SAVAGEWIND_VK_VALIDATION = '1'
        $env:VK_ADD_LAYER_PATH = 'C:\msys64\ucrt64\bin'
        $env:VK_LAYER_VALIDATE_SYNC = '1'
    }
    $effectiveShotAt = if ($ScreenshotAt) { $ScreenshotAt } elseif ($Validation) { $ValidationShotAt } else { 0 }
    if ($effectiveShotAt) { $env:SAVAGEWIND_SHOT_AT = [string]$effectiveShotAt }
    if ($ExtraShots) { $env:SAVAGEWIND_SHOT_AT = $ExtraShots }
    # Keep every measured binary, not only sampled builds. A recorded hash alone
    # cannot reproduce an overwritten executable after a baseline discrepancy.
    # All copying/hashing finishes before launching the game.
    $binarySnapshots = @{}
    foreach ($binary in @($exePath, $env:SAVAGEWIND_GAME_LIB)) {
        $hash = (Get-FileHash -LiteralPath $binary).Hash
        $snapshotDir = Join-Path $repo ("build/codex-perf/binaries/" + $hash)
        $snapshot = Join-Path $snapshotDir ([IO.Path]::GetFileName($binary))
        if (-not (Test-Path -LiteralPath $snapshot)) {
            New-Item -ItemType Directory -Path $snapshotDir -Force | Out-Null
            Copy-Item -LiteralPath $binary -Destination $snapshot
        }
        if ((Get-FileHash -LiteralPath $snapshot).Hash -ne $hash) { throw "Binary snapshot hash mismatch: $snapshot" }
        $binarySnapshots[$binary] = @{hash=$hash; path=$snapshot}
    }
    $manifest = [ordered]@{
        executable=$exePath; executable_sha256=$binarySnapshots[$exePath].hash
        executable_snapshot=$binarySnapshots[$exePath].path
        game_library_sha256=$binarySnapshots[$env:SAVAGEWIND_GAME_LIB].hash
        game_library_snapshot=$binarySnapshots[$env:SAVAGEWIND_GAME_LIB].path
        game_library=$env:SAVAGEWIND_GAME_LIB
        benchmark_script_sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash
        game_files=$gamePath; save_template=$savePath
        settings=(Get-Content -LiteralPath (Join-Path $repo 'savagewind.ini') -Raw)
        seconds=$Seconds; survey=$Survey; svc_profile=[bool]$SvcProfile; diagnostic=([bool]$Diagnostic -or [bool]$Survey); validation=[bool]$Validation
        process_priority=$ProcessPriority
        validation_shot_at=$ValidationShotAt
        screenshot_at=$effectiveShotAt
        legacy_gate=[bool]$LegacyGate; full_barriers=[bool]$FullBarriers
        direct_wake=[bool]$DirectWake; scoped_barriers=[bool]$ScopedBarriers; geometry_gate=[bool]$GeometryGate
        frame_trace=[bool]$FrameTrace; trace_at=$TraceAt; list_stats=[bool]$ListStats; no_shader_warm=[bool]$NoShaderWarm
        draw_batch_probe=[bool]$DrawBatchProbe
        gpu_trace=[bool]$GpuTrace; nsight_exe=$NsightExe
        gpu_trace_start_ms=$GpuTraceStartMs; gpu_trace_duration_ms=$GpuTraceDurationMs
        gpu_trace_frames=$GpuTraceFrames; gpu_trace_shader_profiler=[bool]$GpuTraceShaderProfiler
        pipeline_lto=[bool]$PipelineLto; monolithic_pipelines=[bool]$MonolithicPipelines
        pipeline_stats=[bool]$PipelineStats
        trace_start_s=$TraceStart; trace_count=$TraceCount; trace_delta_ms=$TraceDeltaMs; trace_ratio=$TraceRatio
        trace_large_ms=$TraceLargeMs; trace_large_ratio=$TraceLargeRatio; trace_large_count=$TraceLargeCount
        native_packets=[bool]$NativePackets
        native_multi_draw=$NativeMultiDraw
        native_batch_limit=$NativeBatchLimit
        native_multi_draw_ab_at=$NativeMultiDrawABAt
        early_fence=[bool]$EarlyFence
        safe_overlap=[bool]$SafeOverlap
        readback_dump=[bool]$ReadbackDump
        result_delivery_coverage=[bool]$ResultDeliveryCoverage
        consumer_probe=$ConsumerProbe
        consumer_verify=[bool]$ConsumerVerify
        clock_probe_ms=$ClockProbeMs
        verify_dependencies=([bool]$VerifyDependencies -or [bool]$Validation)
        address_wait=[bool]$AddressWait
        shader_constants=[bool]$ShaderConstants
        sync_probe=[bool]$SyncProbe
        draw_profile=[bool]$DrawProfile; feedback_probe=[bool]$FeedbackProbe
        synchronous_trace_log=[bool]$SynchronousTraceLog
        bounded_constants=[bool]$BoundedConstants
        bounded_constants_bind_full=[bool]$BoundedConstantsBindFull; uniform_copy_audit=[bool]$UniformCopyAudit
        uniform_copy_audit_stride=$UniformCopyAuditStride
        buffer_cache=[bool]$BufferCache
        buffer_cache_ab=[bool]$BufferCacheAB
        buffer_cache_audit=[bool]$BufferCacheAudit
        a1_attribution=[bool]$A1Attribution
        b2=$B2; b2_ab=[bool]$B2AB
        extra_shots=$ExtraShots
        compute_cache=$ComputeCache
        compute_optimize=$ComputeOptimize
        compute_test_limit=$ComputeTestLimit
        range_queries=$RangeQueries
        host_uploads=$HostUploads
        batch_stats=[bool]$BatchStats
        texture_bindings=$TextureBindings
        stage_reuse=$StageReuse
        producer_material=$ProducerMaterial
        producer_material_ab_frames=$ProducerMaterialABFrames
        producer_material_ab_start=$ProducerMaterialABStart
        texture_hash_lanes=$TextureHashLanes
        sparse_write_watch=[bool]$SparseWriteWatch
        write_watch_probe=[bool]$WriteWatchProbe; write_watch_gap_words=$WriteWatchGapWords
        geometry_scan_probe=[bool]$GeometryScanProbe
        geometry_flush_probe=[bool]$GeometryFlushProbe
        geometry_copy_all=$GeometryCopyAll
        geometry_guard=$GeometryGuard
        parallel_watch_probe=[bool]$ParallelWatchProbe
        write_watch_workers=$WriteWatchWorkers
        texture_holder_age=[bool]$TextureHolderAge
        texture_lifetime_probe=[bool]$TextureLifetimeProbe
        compact_records=$CompactRecords
        depth_mirror=[bool]$DepthMirror
        verify_depth_mirror=[bool]$VerifyDepthMirror
        descriptor_workers=$DescriptorWorkers
        cube_layers=[bool]$CubeLayers
        cube_layers_ab=[bool]$CubeLayersAB
        probe_small_pass=$ProbeSmallPass
        verify_cube_layers=$VerifyCubeLayers
        sample=$Sample
        sample_host_only=[bool]$SampleHostOnly; sample_handles=$SampleHandles; sample_buffer_mb=$SampleBufferMB
        input_after_load=$InputAfterLoad
        reboot_at=$RebootAt
        hidden_window=$true
        started=(Get-Date -Format o)
    }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $run 'manifest.json')
    $started = (Get-Date).AddSeconds(-1)
    if ($GpuTrace) {
        $nsightPath = (Resolve-Path -LiteralPath $NsightExe).Path
        # An administratively extracted Nsight has no system-wide layer registry
        # entry. Discover its signed target manifest and enable it for this run.
        $nsightRoot = Split-Path (Split-Path (Split-Path $nsightPath))
        $layerDir = Join-Path $nsightRoot 'target/windows-desktop-nomad-x64'
        $layerManifest = @(Get-ChildItem -LiteralPath $layerDir -Filter 'VK_LAYER_NV_GPU_Trace_*.json')
        if ($layerManifest.Count -ne 1) { throw 'Expected one Nsight GPU Trace Vulkan layer manifest.' }
        $layer = Get-Content -LiteralPath $layerManifest[0].FullName -Raw | ConvertFrom-Json
        $env:VK_ADD_LAYER_PATH = $layerDir
        $env:VK_INSTANCE_LAYERS = $layer.layer.name
        $env:PATH = $layerDir + ';' + $env:PATH
        $captureDir = Join-Path $run 'gpu-trace'
        New-Item -ItemType Directory -Path $captureDir | Out-Null
        $traceArgs = @('--activity="GPU Trace Profiler"', '--platform="Windows (x86_64)"',
            ('--exe="'+$exePath+'"'), ('--dir="'+$repo+'"'), ('--args="\"'+$gamePath+'\" --play"'),
            ('--output-dir="'+$captureDir+'"'), ('--start-after-ms='+$GpuTraceStartMs),
            ('--max-duration-ms='+$GpuTraceDurationMs), ('--limit-to-frames='+$GpuTraceFrames),
            '--architecture="Blackwell GB20x"', '--metric-set-name="Top-Level Triage"',
            '--allocated-timestamps=500000', '--set-gpu-clocks=unaltered', '--auto-export', '--verbose')
        if ($GpuTraceShaderProfiler) { $traceArgs += '--real-time-shader-profiler' }
        $p = Start-Process -FilePath $nsightPath -WorkingDirectory $repo -ArgumentList $traceArgs -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $run 'stdout.log') -RedirectStandardError (Join-Path $run 'stderr.log')
    } else {
        $p = Start-Process -FilePath $exePath -WorkingDirectory $repo -ArgumentList @('"'+$gamePath+'"','--play') -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $run 'stdout.log') -RedirectStandardError (Join-Path $run 'stderr.log')
    }
    $deadline = (Get-Date).AddSeconds($Seconds)
    $priorityApplied = @{}
    do {
        Start-Sleep -Milliseconds 500
        # Startup may relaunch after an address-space collision. Follow only this
        # executable's newly started instances, never an unrelated copy.
        $ours = @(Get-Process savagewind -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -ge $started -and $_.Path -eq $exePath })
        # A run-local scheduling probe, including startup relaunches. Never change
        # another application or persist an OS priority setting.
        if ($ProcessPriority -ne 'Normal') {
            foreach ($owned in $ours) {
                if (-not $priorityApplied.ContainsKey($owned.Id)) {
                    $owned.PriorityClass = [Diagnostics.ProcessPriorityClass]::$ProcessPriority
                    $priorityApplied[$owned.Id] = $true
                }
            }
        }
        if ($GpuTrace -and $p.HasExited) { break }
    } while ((Get-Date) -lt $deadline -and ($ours.Count -or ($GpuTrace -and -not $p.HasExited)))
    if ($ours.Count) {
        # Capture process CPU time before stopping: idle spin removal can reduce
        # host work even when the frame's critical path is unchanged.
        $usage = @($ours | ForEach-Object {
            $_.Refresh()
            [ordered]@{ pid=$_.Id; cpu_seconds=$_.TotalProcessorTime.TotalSeconds
                elapsed_seconds=((Get-Date)-$_.StartTime).TotalSeconds
                peak_working_set_bytes=$_.PeakWorkingSet64 }
        })
        ConvertTo-Json -InputObject $usage | Set-Content -LiteralPath (Join-Path $run 'process-usage.json')
        foreach ($process in $ours) { Stop-Process -Id $process.Id; $process.WaitForExit() }
    } else { Write-Warning 'Game exited before the test deadline; inspect stderr.log.' }
    if ($GpuTrace) {
        $timedOut = -not $p.HasExited
        if ($timedOut) { Stop-Process -Id $p.Id; $p.WaitForExit() }
        [ordered]@{ exit_code=$p.ExitCode; timed_out=$timedOut
            artifacts=@(Get-ChildItem -LiteralPath $captureDir -File -Recurse | Select-Object FullName,Length)
        } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $run 'gpu-trace-result.json')
        if ($timedOut -or $p.ExitCode -ne 0) { throw 'Nsight capture did not complete; inspect stdout.log and gpu-trace-result.json.' }
    }
    if ($effectiveShotAt) {
        $shot = Join-Path $repo ('build/frames/frame_{0:D5}.png' -f $effectiveShotAt)
        if ((Test-Path -LiteralPath $shot) -and (Get-Item -LiteralPath $shot).LastWriteTime -ge $started) {
            Copy-Item -LiteralPath $shot -Destination (Join-Path $run 'render.png')
        }
    }
    if ($NativeMultiDrawABAt) {
        for ($frame=$NativeMultiDrawABAt; $frame -lt $NativeMultiDrawABAt+4; $frame++) {
            $tag=if($frame % 2){'native'}else{'reference'}
            $shot=Join-Path $repo ('build/frames/native_ab_{0:D5}_{1}.png' -f $frame,$tag)
            if ((Test-Path -LiteralPath $shot) -and (Get-Item -LiteralPath $shot).LastWriteTime -ge $started) {
                Copy-Item -LiteralPath $shot -Destination $run
            }
        }
    }
    if ($ReadbackDump) {
        if ($ExtraShots) {
            Get-ChildItem -LiteralPath (Join-Path $repo 'build/frames') -Filter 'frame_*.png' | Where-Object { $_.LastWriteTime -ge $started } | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $run }
        }
        $dumpDir = Join-Path $run 'readbacks'
        New-Item -ItemType Directory -Path $dumpDir | Out-Null
        Get-ChildItem -LiteralPath (Join-Path $repo 'build') -Filter 'linrt_*.raw' | Where-Object { $_.LastWriteTime -ge $started } | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $dumpDir
        }
    }
    if ($Sample) {
        $samplePath = Join-Path $repo 'build/samples.txt'
        if ((Test-Path -LiteralPath $samplePath) -and (Get-Item -LiteralPath $samplePath).LastWriteTime -ge $started) {
            Copy-Item -LiteralPath $samplePath -Destination (Join-Path $run 'samples.txt')
        }
    }
    Write-Output $run
} finally {
    Get-ChildItem Env:SAVAGEWIND_* | Remove-Item
    foreach ($key in $backup.Keys) { Set-Item -LiteralPath "Env:$key" -Value $backup[$key] }
    foreach ($key in $vkBackup.Keys) {
        if ($null -eq $vkBackup[$key]) { Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue }
        else { Set-Item -LiteralPath "Env:$key" -Value $vkBackup[$key] }
    }
    $env:PATH = $oldPath
}
