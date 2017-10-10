use glutin::{ElementState, MouseButton, MouseScrollDelta};
pub use glutin::VirtualKeyCode as Key;
use mint;

use std::collections::HashSet;
use std::time;

mod timer;
pub mod axis;

pub use self::axis::{AXIS_DOWN_UP, AXIS_LEFT_RIGHT};

pub use self::timer::Timer;

const PIXELS_PER_LINE: f32 = 38.0;

pub type TimerDuration = f32;

// TODO: Remove
struct State {
    time_moment: time::Instant,
    is_focused: bool,
    keys_pressed: HashSet<Key>,
    mouse_pressed: HashSet<MouseButton>,
    mouse_pos: mint::Point2<f32>,
    mouse_pos_ndc: mint::Point2<f32>,
}

struct Delta {
    time_delta: TimerDuration,
    keys_hit: Vec<Key>,
    mouse_moves: Vec<mint::Vector2<f32>>,
    mouse_moves_ndc: Vec<mint::Vector2<f32>>,
    axes_raw: Vec<(u8, f32)>,
    mouse_hit: Vec<MouseButton>,
    mouse_wheel: Vec<f32>,
}

/// Controls user and system input from keyboard, mouse and system clock.
pub struct Input {
    state: State,
    delta: Delta,
}

impl Input {
    pub(crate) fn new() -> Self {
        let state = State {
            time_moment: time::Instant::now(),
            is_focused: true,
            keys_pressed: HashSet::new(),
            mouse_pressed: HashSet::new(),
            mouse_pos: [0.0; 2].into(),
            mouse_pos_ndc: [0.0; 2].into(),
        };
        let delta = Delta {
            time_delta: 0.0,
            keys_hit: Vec::new(),
            mouse_moves: Vec::new(),
            mouse_moves_ndc: Vec::new(),
            axes_raw: Vec::new(),
            mouse_hit: Vec::new(),
            mouse_wheel: Vec::new(),
        };
        Input { state, delta }
    }

    pub fn reset(&mut self) {
        let now = time::Instant::now();
        let dt = now - self.state.time_moment;
        self.state.time_moment = now;
        self.delta.time_delta = dt.as_secs() as f32 + 1e-9 * dt.subsec_nanos() as f32;
        self.delta.keys_hit.clear();
        self.delta.mouse_moves.clear();
        self.delta.mouse_moves_ndc.clear();
        self.delta.axes_raw.clear();
        self.delta.mouse_hit.clear();
        self.delta.mouse_wheel.clear();
    }

    /// Create new timer.
    pub fn time(&self) -> Timer {
        Timer {
            start: self.state.time_moment,
        }
    }

    /// Get current delta time (time since previous frame) in seconds.
    pub fn delta_time(&self) -> f32 {
        self.delta.time_delta
    }

    /// Get current mouse pointer position in pixels from top-left
    pub fn mouse_pos(&self) -> mint::Point2<f32> {
        self.state.mouse_pos
    }

    /// Get current mouse pointer position in Normalized Display Coordinates.
    /// See [`map_to_ndc`](struct.Renderer.html#method.map_to_ndc).
    pub fn mouse_pos_ndc(&self) -> mint::Point2<f32> {
        self.state.mouse_pos_ndc
    }

    /// Get list of all mouse wheel movements since last frame.
    pub fn mouse_wheel_movements(&self) -> &[f32] {
        &self.delta.mouse_wheel[..]
    }

    /// Get summarized mouse wheel movement (the sum of all movements since last frame).
    pub fn mouse_wheel(&self) -> f32 {
        self.delta.mouse_wheel.iter().sum()
    }

    /// Get list of all mouse movements since last frame in pixels.
    pub fn mouse_movements(&self) -> &[mint::Vector2<f32>] {
        &self.delta.mouse_moves[..]
    }

    /// Get list of all mouse movements since last frame in NDC.
    pub fn mouse_movements_ndc(&self) -> &[mint::Vector2<f32>] {
        &self.delta.mouse_moves_ndc[..]
    }

    /// Get list of all raw inputs since last frame. It usually corresponds to mouse movements.
    pub fn axes_movements_raw(&self) -> &[(u8, f32)] {
        &self.delta.axes_raw[..]
    }

    fn calculate_delta(moves: &[mint::Vector2<f32>]) -> mint::Vector2<f32> {
        use cgmath::Vector2;
        moves
            .iter()
            .cloned()
            .map(Vector2::from)
            .sum::<Vector2<f32>>()
            .into()
    }

    /// Get summarized mouse movements (the sum of all movements since last frame) in pixels.
    pub fn mouse_delta(&self) -> mint::Vector2<f32> {
        Input::calculate_delta(self.mouse_movements())
    }

    /// Get summarized mouse movements (the sum of all movements since last frame) in NDC.
    pub fn mouse_delta_ndc(&self) -> mint::Vector2<f32> {
        Input::calculate_delta(self.mouse_movements_ndc())
    }

    /// Get summarized raw input since last frame. It usually corresponds to mouse movements.
    pub fn mouse_delta_raw(&self) -> mint::Vector2<f32> {
        use cgmath::Vector2;
        self.delta
            .axes_raw
            .iter()
            .filter(|&&(axis, _)| axis == 0 || axis == 1)
            .map(|&(axis, value)| if axis == 0 {
                (value, 0.0)
            } else {
                (0.0, value)
            })
            .map(|t| Vector2 { x: t.0, y: t.1 })
            .sum::<Vector2<f32>>()
            .into()
    }

    pub(crate) fn window_focus(
        &mut self,
        state: bool,
    ) {
        self.state.is_focused = state;
    }

    pub(crate) fn keyboard_input(
        &mut self,
        state: ElementState,
        key: Key,
    ) {
        match state {
            ElementState::Pressed => {
                self.state.keys_pressed.insert(key);
                self.delta.keys_hit.push(key);
            }
            ElementState::Released => {
                self.state.keys_pressed.remove(&key);
            }
        }
    }

    pub(crate) fn mouse_input(
        &mut self,
        state: ElementState,
        button: MouseButton,
    ) {
        match state {
            ElementState::Pressed => {
                self.state.mouse_pressed.insert(button);
                self.delta.mouse_hit.push(button);
            }
            ElementState::Released => {
                self.state.mouse_pressed.remove(&button);
            }
        }
    }

    pub(crate) fn mouse_moved(
        &mut self,
        pos: mint::Point2<f32>,
        pos_ndc: mint::Point2<f32>,
    ) {
        use cgmath::Point2;
        self.delta
            .mouse_moves
            .push((Point2::from(pos) - Point2::from(self.state.mouse_pos)).into());
        self.delta
            .mouse_moves_ndc
            .push((Point2::from(pos_ndc) - Point2::from(self.state.mouse_pos_ndc)).into());
        self.state.mouse_pos = pos;
        self.state.mouse_pos_ndc = pos_ndc;
    }

    pub(crate) fn axis_moved_raw(
        &mut self,
        axis: u8,
        value: f32,
    ) {
        println!("Axis moved raw: {} {}", axis, value);
        self.delta.axes_raw.push((axis, value));
    }

    pub(crate) fn mouse_wheel_input(
        &mut self,
        delta: MouseScrollDelta,
    ) {
        self.delta.mouse_wheel.push(match delta {
            MouseScrollDelta::LineDelta(_, y) => y * PIXELS_PER_LINE,
            MouseScrollDelta::PixelDelta(_, y) => y,
        });
    }

    pub fn button(
        &self,
        button: Button,
    ) -> ButtonInfo {
        use std::u8::MAX;
        let pressed = match button {
            Button::Key(button) => self.state.keys_pressed.contains(&button),
            Button::Mouse(button) => self.state.mouse_pressed.contains(&button),
        };
        let released = !pressed;
        let hits = match button {
            Button::Key(button) => self.delta
                .keys_hit
                .iter()
                .filter(|&&key| key == button)
                .take(MAX as usize)
                .count() as u8,
            Button::Mouse(button) => self.delta
                .mouse_hit
                .iter()
                .filter(|&&key| key == button)
                .take(MAX as usize)
                .count() as u8,
        };
        ButtonInfo {
            pressed,
            released,
            hits,
        }
    }

    pub fn key_axis(
        &self,
        axis: axis::Key,
    ) -> KeyAxisInfo {
        let state = {
            let is_pos = self.state.keys_pressed.contains(&axis.pos);
            let is_neg = self.state.keys_pressed.contains(&axis.neg);
            if is_pos && !is_neg {
                Some(1)
            } else if is_neg && !is_pos {
                Some(-1)
            } else {
                None
            }
        };
        let timed_state = state.map(|v| v as f32 * self.delta_time());
        let mut hits = KeyAxisHits::default();
        hits.pos = self.delta
            .keys_hit
            .iter()
            .filter(|&&k| k == axis.pos)
            .count() as u8;
        hits.neg = self.delta
            .keys_hit
            .iter()
            .filter(|&&k| k == axis.neg)
            .count() as u8;
        let delta_hits = if hits.pos + hits.neg == 0 {
            None
        } else {
            Some(hits.pos as i8 - hits.neg as i8)
        };
        KeyAxisInfo {
            state,
            timed_state,
            hits,
            delta_hits,
        }
    }

    pub fn axis(
        &self,
        axis: axis::Raw,
    ) -> RawAxisInfo {
        let delta = {
            let moves = self.delta
                .axes_raw
                .iter()
                .filter(|&&(id, _)| id == axis.id)
                .map(|&(_, value)| value)
                .collect::<Vec<_>>();
            if moves.len() == 0 {
                None
            } else {
                Some(moves.iter().sum::<f32>())
            }
        };
        RawAxisInfo { delta }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct ButtonInfo {
    pub pressed: bool,
    pub released: bool,
    pub hits: u8,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct KeyAxisHits {
    pos: u8,
    neg: u8,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct KeyAxisInfo {
    pub state: Option<i8>,
    pub timed_state: Option<f32>,
    pub hits: KeyAxisHits,
    pub delta_hits: Option<i8>,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct RawAxisInfo {
    pub delta: Option<f32>,
}

/// Keyboard or mouse button.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Button {
    /// Keyboard button.
    Key(Key),
    /// Mouse button.
    Mouse(MouseButton),
}

/// `Escape` keyboard button.
pub const KEY_ESCAPE: Button = Button::Key(Key::Escape);
/// `Space` keyboard button.
pub const KEY_SPACE: Button = Button::Key(Key::Space);
/// Left mouse button.
pub const MOUSE_LEFT: Button = Button::Mouse(MouseButton::Left);
/// Right mouse button.
pub const MOUSE_RIGHT: Button = Button::Mouse(MouseButton::Right);
