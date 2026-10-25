#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <shaderc/shaderc.h>
#include "../runtime/nvn/vulkan/native_draw_constants.h"

int main(int argc, char **argv) {
    const char *decl="#version 460\nlayout(std140, set = 1, binding = 3) uniform CB3 { uvec4 cb3[8]; };\n";
    char source[2048]; uint32_t mask=123;
    snprintf(source,sizeof source,"%suint f(){return cb3[7][3]^cbr(cb3,(4u+8u));}\n",decl);
    char *out=vk_shader_native_draw_constants(source,1,"0u",&mask);
    assert(out&&mask==8&&!strstr(out,"uniform CB3")&&strstr(out,"constants[21]")&&strstr(out,"sw_native_cb3(7u)[3]")&&strstr(out,"sw_native_cbr3((4u+8u))"));free(out);
    snprintf(source,sizeof source,"%s// pc cb0: not a load\n/* cb2[0][0] */uint f(){return cb3[0][0];}\n",decl);
    out=vk_shader_native_draw_constants(source,1,"0u",&mask);
    assert(out&&strstr(out,"// pc cb0: not a load")&&strstr(out,"/* cb2[0][0] */"));free(out);
    const char *bad[]={"cb3[8][0]","cb3[4294967296][0]","cb3[i][0]","cb3","other(cb3)","cb3.length()","cbr(cb4,0u)","cb4[0][0]"};
    for(unsigned i=0;i<sizeof bad/sizeof *bad;i++){
        snprintf(source,sizeof source,"%suint f(){return %s;}\n",decl,bad[i]);
        assert(!vk_shader_native_draw_constants(source,1,"0u",&mask)&&mask==0);
    }
    assert(!vk_shader_native_draw_constants(decl,0,"0u",NULL));
    assert(!vk_shader_native_draw_constants(decl,2,"0u",NULL));
    assert(!vk_shader_native_draw_constants(NULL,0,"0u",NULL));
    assert(!vk_shader_native_draw_constants(decl,1,"",NULL));
    assert(!vk_shader_native_draw_constants("#version 460\nvoid main(){}",1,"0u",NULL));
    snprintf(source,sizeof source,"%svoid main() {\nuint value=cb3[0][0];}\n",decl);
    out=vk_shader_native_draw_graphics(source,1,31,&mask);
    assert(out&&mask==8&&strstr(out,"layout(location = 31) flat in uint sw_batch_draw_id")&&strstr(out,"draws[sw_batch_draw_id]"));free(out);
    const char *simple="#version 460\nvoid main() {\ngl_Position=vec4(0,0,0,1);}\n";
    out=vk_shader_native_draw_graphics(simple,0,31,&mask);
    assert(out&&mask==0&&strstr(out,"sw_batch_draw_id = uint(gl_DrawID) + sw_batch_root.first"));free(out);
    assert(!vk_shader_native_draw_graphics(simple,0,32,&mask));
    assert(!vk_shader_native_draw_graphics("#version 460\nlayout(location = 31) out vec4 existing;\nvoid main() {\n}\n",0,31,&mask));
    puts("native draw constant parser: static/dynamic loads, stage indexing, bounds and fail-closed syntax passed");
    if(argc!=2)return 0;
    DIR *dir=opendir(argv[1]);assert(dir);struct dirent *entry;
    shaderc_compiler_t compiler=shaderc_compiler_initialize();shaderc_compile_options_t options=shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(options,shaderc_target_env_vulkan,shaderc_env_version_vulkan_1_3);
    shaderc_compile_options_set_optimization_level(options,shaderc_optimization_level_performance);
    unsigned files=0,rewritten=0,rejected=0;uint64_t hashes[2048];unsigned unique=0;
    while((entry=readdir(dir))){
        size_t n=strlen(entry->d_name);if(n<8)continue;
        int stage=!strcmp(entry->d_name+n-8,"_vs.glsl")?0:!strcmp(entry->d_name+n-8,"_fs.glsl")?1:-1;
        if(stage<0)continue;
        char path[2048];snprintf(path,sizeof path,"%s/%s",argv[1],entry->d_name);
        FILE *f=fopen(path,"rb");assert(f);fseek(f,0,SEEK_END);long size=ftell(f);rewind(f);assert(size>0);
        char *text=malloc((size_t)size+1);assert(text);assert(fread(text,1,(size_t)size,f)==(size_t)size);fclose(f);text[size]=0;
        char *write=text;for(char *read=text;*read;read++){if(*read=='\r'&&read[1]=='\n')continue;*write++=*read;}*write=0;
        files++;uint64_t h=1469598103934665603ull;for(char *p=text;*p;p++)h=(h^(unsigned char)*p)*1099511628211ull;
        bool duplicate=false;for(unsigned i=0;i<unique;i++)duplicate|=hashes[i]==h;
        if(duplicate){free(text);continue;}assert(unique<2048);hashes[unique++]=h;
        out=NULL;
        // This is an individual-shader compile test. Runtime pairing must find
        // one shared free location and include builtins in device output limits.
        for(unsigned location=0;location<28&&!out;location++)
            out=vk_shader_native_draw_graphics(text,(unsigned)stage,location,&mask);
        if(!out){rejected++;printf("fallback %s\n",entry->d_name);free(text);continue;}
        for(unsigned variant=0;variant<2;variant++){
            const char *input=variant?out:text;
            shaderc_compilation_result_t compiled=shaderc_compile_into_spv(compiler,input,strlen(input),stage?shaderc_fragment_shader:shaderc_vertex_shader,path,"main",options);
            if(shaderc_result_get_compilation_status(compiled)!=shaderc_compilation_status_success){fprintf(stderr,"variant %u %s\n%s\n",variant,path,shaderc_result_get_error_message(compiled));return 1;}
            shaderc_result_release(compiled);
        }
        rewritten++;free(out);free(text);
    }
    closedir(dir);shaderc_compile_options_release(options);shaderc_compiler_release(compiler);
    printf("native draw graphics GLSL corpus: %u files, %u unique sources, %u original/candidate compile pairs, %u conservative fallbacks. Compilation is not image equivalence or FPS evidence.\n",files,unique,rewritten,rejected);
    return 0;
}
