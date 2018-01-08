extern crate three;

use three::Object;

fn main() {
    let mut win = three::Window::new("Three-rs obj loading example");
    let cam = win.factory.perspective_camera(60.0, 1.0 .. 10.0);

    while win.update() && !win.input.hit(three::KEY_ESCAPE) {
        win.render(&cam);
    }
}
