#include <assert.h>
#include <shaderc/shaderc.h>
#include "../runtime/nvn/vulkan/buffer_cache_shader.h"
int main(int argc, char **argv) {
    assert(!buffer_cache_shader("void main() { atomicAdd(gmem_.w[0],1); }"));
    assert(!buffer_cache_shader("void main() { gmem_.w[0] += 1; }"));
    assert(!buffer_cache_shader("void main() { gmem_.w[0] = f(); }"));
    shaderc_compiler_t compiler=shaderc_compiler_initialize();
    shaderc_compile_options_t options=shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(options,shaderc_target_env_vulkan,shaderc_env_version_vulkan_1_3);
    int accepted=0;
    for(int i=1;i<argc;i++) {
        FILE *f=fopen(argv[i],"rb");assert(f);fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);
        char *src=malloc(n+1);assert(src&&fread(src,1,n,f)==(size_t)n);fclose(f);src[n]=0;
        int stage=strstr(src,"gl_Position")?0:strstr(src,"local_size")?5:1;
        if(strstr(argv[i],"c3e522014faa8ebd")) {
            assert(buffer_cache_graphics_proved(src,stage));
            char *crlf=malloc(2*n+1);size_t k=0;
            for(long j=0;j<n;j++){if(src[j]=='\r')continue;if(src[j]=='\n')crlf[k++]='\r';crlf[k++]=src[j];}crlf[k]=0;
            assert(buffer_cache_graphics_proved(crlf,stage));free(crlf);
        }
        char *rewritten=buffer_cache_shader_stage(src,stage);
        if(!rewritten){printf("fallback %s\n",argv[i]);free(src);continue;}
        assert(!strstr(strstr(rewritten,"void main()"),"gmem_.w["));
        assert(!buffer_cache_shader(rewritten));
        shaderc_shader_kind kind=stage==0?shaderc_vertex_shader:stage==5?shaderc_compute_shader:shaderc_fragment_shader;
        shaderc_compilation_result_t r=shaderc_compile_into_spv(compiler,rewritten,strlen(rewritten),kind,argv[i],"main",options);
        if(shaderc_result_get_compilation_status(r)!=shaderc_compilation_status_success){fprintf(stderr,"%s\n",shaderc_result_get_error_message(r));return 1;}
        printf("compiled %s graphics_proof=%d\n",argv[i],buffer_cache_graphics_proved(src,stage));accepted++;shaderc_result_release(r);free(src);free(rewritten);
    }
    shaderc_compile_options_release(options);shaderc_compiler_release(compiler);
    printf("accepted %d/%d; unsupported forms reject\n",accepted,argc-1);return 0;
}
