#version 150 core
#include <lights>
#include <globals>

in vec4 a_Position;
in vec4 a_Normal;
out vec3 v_World;
out vec3 v_Normal;
out vec3 v_Half[MAX_LIGHTS];
out vec4 v_ShadowCoord[MAX_LIGHTS];

in vec4 a_World0;
in vec4 a_World1;
in vec4 a_World2;
in vec4 a_MatParams;
in vec4 a_Color;
in vec4 a_UvRange;

void main() {
    mat4 m_World = mat4(a_World0, a_World1, a_World2, vec4(0.0, 0.0, 0.0, 1.0));
    vec4 world = m_World * a_Position;
    v_World = world.xyz;
    v_Normal = normalize(mat3(m_World) * a_Normal.xyz);
    for(uint i=0U; i < min(MAX_LIGHTS, u_NumLights); ++i) {
        Light light = u_Lights[i];
        vec3 dir = light.pos.xyz - light.pos.w * world.xyz;
        v_Half[i] = normalize(v_Normal + normalize(dir));
        v_ShadowCoord[i] = light.projection * world;
    }
    gl_Position = u_ViewProj * world;
}
