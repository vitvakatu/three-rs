use glutin::VirtualKeyCode as KeyCode;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    pub neg: KeyCode,
    pub pos: KeyCode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Raw {
    pub id: u8,
}

/// Axis for left and right arrow keys.
pub const AXIS_LEFT_RIGHT: Key = Key {
    neg: KeyCode::Left,
    pos: KeyCode::Right,
};
/// Axis for up and down arrow keys.
pub const AXIS_DOWN_UP: Key = Key {
    neg: KeyCode::Down,
    pos: KeyCode::Up,
};
