extern crate cgmath;
extern crate mint;
extern crate three;

use cgmath::{Angle, Decomposed, One, Quaternion, Rad, Rotation3, Transform, Vector3};

struct Level {
    speed: f32,
}

struct Cube {
    group: three::Group,
    mesh: three::Mesh,
    level_id: usize,
    orientation: Quaternion<f32>,
}

fn create_cubes(
    factory: &mut three::Factory,
    materials: &[three::material::Lambert],
    levels: &[Level],
) -> Vec<Cube> {
    let mut geometry = three::Geometry::cuboid(2.0, 2.0, 2.0);
    for v in geometry.base_shape.vertices.iter_mut() {
        v.z += 1.0;
    }

    let root = {
        let mut group = factory.group();
        let mut mesh = factory.mesh(geometry.clone(), materials[0].clone());
        group.set_position([0.0, 0.0, 1.0]);
        group.set_scale(2.0);
        mesh.set_parent(&group);
        Cube {
            group,
            mesh,
            level_id: 0,
            orientation: Quaternion::one(),
        }
    };
    let mut list = vec![root];

    struct Stack {
        parent_id: usize,
        mat_id: usize,
        lev_id: usize,
    }
    let mut stack = vec![
        Stack {
            parent_id: 0,
            mat_id: 1,
            lev_id: 1,
        },
    ];

    let axis = [
        Vector3::unit_z(),
        Vector3::unit_x(),
        -Vector3::unit_x(),
        Vector3::unit_y(),
        -Vector3::unit_y(),
    ];
    let children: Vec<_> = axis.iter()
        .map(|&axe| {
            Decomposed {
                disp: Vector3::new(0.0, 0.0, 1.0),
                rot: Quaternion::from_axis_angle(axe, Rad::turn_div_4()),
                scale: 1.0,
            }.concat(&Decomposed {
                disp: Vector3::new(0.0, 0.0, 1.0),
                rot: Quaternion::one(),
                scale: 0.4,
            })
        })
        .collect();

    while let Some(next) = stack.pop() {
        for child in &children {
            let mat = materials[next.mat_id].clone();
            let mut cube = Cube {
                group: factory.group(),
                mesh: factory.mesh_instance_with_material(&list[0].mesh, mat),
                level_id: next.lev_id,
                orientation: child.rot,
            };
            let p: mint::Vector3<f32> = child.disp.into();
            cube.group.set_transform(p, child.rot, child.scale);
            cube.group.set_parent(&list[next.parent_id].group);
            cube.mesh.set_parent(&cube.group);
            if next.mat_id + 1 < materials.len() && next.lev_id + 1 < levels.len() {
                stack.push(Stack {
                    parent_id: list.len(),
                    mat_id: next.mat_id + 1,
                    lev_id: next.lev_id + 1,
                });
            }
            list.push(cube);
        }
    }

    list
}

const COLORS: [three::Color; 6] = [0xffff80, 0x8080ff, 0x80ff80, 0xff8080, 0x80ffff, 0xff80ff];

const SPEEDS: [f32; 6] = [0.7, -1.0, 1.3, -1.6, 1.9, -2.2];

fn main() {
    let mut win = three::Window::new("Three-rs group example");
    win.scene.background = three::Background::Color(0x204060);

    let mut cam = win.factory.perspective_camera(60.0, 1.0 .. 100.0);
    cam.look_at([-1.8, -8.0, 7.0], [0.0, 0.0, 3.5], None);

    let mut light = win.factory.point_light(0xffffff, 1.0);
    light.set_position([0.0, -10.0, 10.0]);
    light.set_parent(&win.scene);

    let materials: Vec<_> = COLORS
        .iter()
        .map(|&color| three::material::Lambert { color, flat: false })
        .collect();
    let levels: Vec<_> = SPEEDS.iter().map(|&speed| Level { speed }).collect();
    let mut cubes = create_cubes(&mut win.factory, &materials, &levels);
    cubes[0].group.set_parent(&win.scene);

    let font = win.factory.load_font(format!(
        "{}/data/fonts/DejaVuSans.ttf",
        env!("CARGO_MANIFEST_DIR")
    ));
    let mut fps_counter = win.factory.ui_text(&font, "FPS: 00");

    let timer = win.input.time();
    while win.update() && !win.input.hit(three::KEY_ESCAPE) {
        let time = timer.get(&win.input);
        let delta_time = win.input.delta_time();
        fps_counter.set_text(format!("FPS: {}", 1.0 / delta_time));
        for cube in cubes.iter_mut() {
            let level = &levels[cube.level_id];
            let angle = Rad(time * level.speed);
            let q = cube.orientation * cgmath::Quaternion::from_angle_z(angle);
            cube.group.set_orientation(q);
        }

        win.render(&cam);
    }
}
