extern crate cgmath;
extern crate mint;
extern crate three;

use cgmath::prelude::*;

fn main() {
    let mut win = three::Window::new("Three-rs shapes example");
    let mut cam = win.factory.perspective_camera(75.0, 1.0 .. 50.0);
    cam.set_position([0.0, 0.0, 10.0]);

    let mut mbox = {
        let geometry = three::Geometry::cuboid(3.0, 2.0, 1.0);
        let material = three::Material::MeshBasic {
            color: 0x00ff00,
            map: None,
            wireframe: true,
        };
        win.factory.mesh(geometry, material)
    };
    mbox.set_position([-3.0, -3.0, 0.0]);
    win.scene.add(&mbox);

    let mut mcyl = {
        let geometry = three::Geometry::cylinder(1.0, 2.0, 2.0, 5);
        let material = three::Material::MeshBasic {
            color: 0xff0000,
            map: None,
            wireframe: true,
        };
        win.factory.mesh(geometry, material)
    };
    mcyl.set_position([3.0, -3.0, 0.0]);
    win.scene.add(&mcyl);

    let mut msphere = {
        let geometry = three::Geometry::uv_sphere(2.0, 5, 5);
        let material = three::Material::MeshBasic {
            color: 0xff0000,
            map: None,
            wireframe: true,
        };
        win.factory.mesh(geometry, material)
    };
    msphere.set_position([-3.0, 3.0, 0.0]);
    win.scene.add(&msphere);

    let mut mline = {
        let geometry = three::Geometry::with_vertices(vec![
            [-2.0, -1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            [2.0, -1.0, 0.0].into(),
        ]);
        let material = three::Material::LineBasic { color: 0x0000ff };
        win.factory.mesh(geometry, material)
    };
    mline.set_position([3.0, 3.0, 0.0]);
    win.scene.add(&mline);

    let mut angle = cgmath::Rad::zero();
    while win.update() && win.input.button(three::KEY_ESCAPE).released {
        if let Some(diff) = win.input.key_axis(three::AXIS_LEFT_RIGHT).timed_state {
            angle += cgmath::Rad(1.5 * diff);
            let q = cgmath::Quaternion::from_angle_y(angle);
            mbox.set_orientation(q);
            mcyl.set_orientation(q);
            mline.set_orientation(q);
            msphere.set_orientation(q);
        }

        win.render(&cam);
    }
}
