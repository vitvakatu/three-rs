#version 150 core

in vec2 v_TexCoord;
out vec4 Target0;

uniform sampler2D t_BlitTex;

void main() {
    Target0 = texture(t_BlitTex, v_TexCoord);
}