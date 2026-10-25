#include <assert.h>
#include "../runtime/nvn/vulkan/shader_store_audit.h"

int main(int argc, char **argv) {
    const char *valid = "void main(){uint r0=gmem_.w[3u]; if(true) gmem_.w[4u]=r0; gmem_.w[5u]=0u;}";
    char *out = shader_store_audit(valid);
    assert(out && strstr(out, "r0=audit_read(3u)") && strstr(out, "audit_store(4u,r0);") && strstr(out, "audit_store(5u,0u);"));
    free(out);
    out = shader_store_audit("void main(){gmem_.w[0u]=r0; r1=gmem_.w[0u];}");
    assert(out && strstr(out, "audit_values[i-1u]") && strstr(out, "r1=audit_read(0u)")); free(out);
    const char *bad[] = {
        "void main(){gmem_.w[0u]=r0; barrier(); r1=gmem_.w[0u];}",
        "void main(){gmem_.w[0u]+=r0;}",
        "void main(){gmem_.w[0u]=r0+1u;}",
        "void main(){atomicAdd(gmem_.w[0u],1u);}",
        "void main(){imageStore(i,p,v); gmem_.w[0u]=r0;}",
        "shared uint s; void main(){gmem_.w[0u]=r0;}",
        "void main(){r0=gmem_.w[3u];}",
        "void main(){gmem_.w[0u=r0;}"
    };
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) assert(!shader_store_audit(bad[i]));
    uint32_t journal[2 * STORE_AUDIT_WORDS] = {0}, other[2 * STORE_AUDIT_WORDS];
    StoreAuditStats stats;
    journal[0] = 1; journal[1] = 1; journal[2] = 7; journal[3] = 42; journal[STORE_AUDIT_READ_BASE] = 7;
    journal[STORE_AUDIT_WORDS] = 1; journal[STORE_AUDIT_WORDS+2] = 8; journal[STORE_AUDIT_WORDS+3] = 43;
    memcpy(other, journal, sizeof journal);
    assert(!store_audit_check(journal, other, 2, &stats) && stats.stores == 2 && stats.reads == 1);
    other[3]++;
    assert(store_audit_check(journal, other, 2, &stats) == 1);
    journal[STORE_AUDIT_READ_BASE] = 8; memcpy(other, journal, sizeof journal);
    assert(store_audit_check(journal, other, 2, &stats) == -5); // other invocation's store
    journal[STORE_AUDIT_WORDS+2] = 7; memcpy(other, journal, sizeof journal);
    assert(store_audit_check(journal, other, 2, &stats) == -4); // write/write overlap
    journal[0] = STORE_AUDIT_MAX + 1;
    assert(store_audit_check(journal, other, 2, &stats) == -1);
    memset(journal, 0, sizeof journal); memset(other, 0, sizeof other);
    assert(store_audit_check(journal, other, 2, &stats) == -2);
    if (argc == 3) {
        FILE *f = fopen(argv[1], "rb"); assert(f);
        fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f); assert(n > 0);
        char *src = malloc((size_t)n + 1); assert(src);
        assert(fread(src, 1, (size_t)n, f) == (size_t)n); fclose(f); src[n] = 0;
        out = shader_store_audit(src); assert(out);
        f = fopen(argv[2], "wb"); assert(f); fputs(out, f); fclose(f);
        free(src); free(out);
    }
    puts("store audit: exact stores, read preservation, unsupported side-effect and syntax rejection passed");
    return 0;
}
