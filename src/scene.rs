//! `Scene` and `SyncGuard` structures.
use color::Color;
use hub::{Hub, HubPtr};
use node::Node;
use object::Object;
use texture::{CubeMap, Texture};

use std::sync::MutexGuard;

/// Unique identifier for a scene.
pub type Uid = usize;

/// Background type.
#[derive(Clone, Debug, PartialEq)]
pub enum Background {
    /// Basic solid color background.
    Color(Color),
    /// Texture background, covers the whole screen.
    // TODO: different wrap modes?
    Texture(Texture<[f32; 4]>),
    /// Skybox
    Skybox(CubeMap<[f32; 4]>),
}

/// The root node of a tree of game objects that may be rendered by a [`Camera`].
///
/// [`Camera`]: ../camera/struct.Camera.html
pub struct Scene {
    pub(crate) object: Object,
    pub(crate) hub: HubPtr,
    /// See [`Background`](struct.Background.html).
    pub background: Background,
}

/// `SyncGuard` is used to obtain information about in scene nodes in most effective way.
///
/// # Examples
///
/// Imagine that you have your own helper type `Enemy`:
///
/// ```rust
/// # #[macro_use]
/// # extern crate three;
/// struct Enemy {
///     mesh: three::Mesh,
///     is_visible: bool,
/// }
/// three_object_wrapper!(Enemy::mesh);
/// # fn main() {}
/// ```
///
/// You need this wrapper around `three::Mesh` to cache some information - in our case, visibility.
///
/// In your game you contain all your enemy objects in `Vec<Enemy>`. In the main loop you need
/// to iterate over all the enemies and make them visible or not, basing on current position.
/// The most obvious way is to use [`Object::sync`], but it's not the best idea from the side of
/// performance. Instead, you can create `SyncGuard` and use its `update` method to effectively
/// walk through every enemy in your game:
///
/// ```rust,no_run
/// # #[macro_use]
/// # extern crate three;
/// # #[derive(Clone)]
/// # struct Enemy {
/// #     mesh: three::Mesh,
/// #     is_visible: bool,
/// # }
/// # three_object_wrapper!(Enemy::mesh);
/// # fn main() {
/// # let mut win = three::Window::new("SyncGuard example");
/// # let geometry = three::Geometry::default();
/// # let material = three::material::Basic { color: three::color::RED, map: None };
/// # let mesh = win.factory.mesh(geometry, material);
/// # let mut enemy = Enemy { mesh, is_visible: true };
/// # enemy.set_parent(&win.scene);
/// # let mut enemies = vec![enemy];
/// # while true {
/// let mut sync = win.scene.sync_guard();
/// sync.update(enemies.iter_mut(), |enemy, node| {
///     let position = node.world_transform.position;
///     if position.x > 10.0 {
///         enemy.is_visible = false;
///         enemy.set_visible(false);
///     } else {
///         enemy.is_visible = true;
///         enemy.set_visible(true);
///     }
/// });
/// # }}
/// ```
///
/// [`Object::sync`]: ../struct.Object.html#method.sync
pub struct SyncGuard<'a> {
    hub: MutexGuard<'a, Hub>,
    scene_id: Option<Uid>,
}

impl<'a> SyncGuard<'a> {
    /// Walk through every item in `iterator`, obtain its [`Node`] and call user-supplied closure.
    ///
    /// # Panics
    /// Panics if `scene` doesn't have this `Object`.
    ///
    /// [`Node`]: ../node/struct.Node.html
    pub fn update<T: 'a, I, F>(
        &mut self,
        iterator: I,
        mut closure: F,
    ) where
        T: AsRef<Object>,
        I: Iterator<Item = &'a mut T>,
        F: FnMut(&'a mut T, Node),
    {
        iterator.for_each(|item| {
            let node = {
                let object: &Object = item.as_ref();
                let node = &self.hub.nodes[&object.node];
                assert_eq!(node.scene_id, self.scene_id);
                node.into()
            };
            closure(item, node);
        })
    }
}

impl Scene {
    /// Create new [`SyncGuard`](struct.SyncGuard.html).
    ///
    /// This is performance-costly operation, you should not use it many times per frame.
    pub fn sync_guard<'a>(&'a mut self) -> SyncGuard<'a> {
        let mut hub = self.hub.lock().unwrap();
        hub.process_messages();
        hub.update_graph();
        let scene_id = hub.nodes[&self.object.node].scene_id;
        SyncGuard { hub, scene_id }
    }
}

three_object_wrapper!(Scene);
