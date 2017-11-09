#version 150 core
#include <globals>

in vec4 a_Position;
in vec4 a_World0;
in vec4 a_World1;
in vec4 a_World2;

void main() {
    mat4 m_World = mat4(a_World0, a_World1, a_World2, vec4(0.0, 0.0, 0.0, 1.0));
    gl_Position = u_ViewProj * m_World * a_Position;
}
