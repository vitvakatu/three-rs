#version 150 core

in vec2 v_TexCoord;
in vec4 a_Color;
out vec4 Target0;

uniform sampler2D t_Map;

void main() {
    Target0 = a_Color * texture(t_Map, v_TexCoord);
}
