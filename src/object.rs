use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::mpsc;

use mint;

use hub::{Message, Operation};
use node::{Node, NodePointer};
use scene::Scene;

//Note: no local state should be here, only remote links
/// `Object` represents an entity that can be added to the scene.
///
/// There is no need to use `Object` directly, there are specific wrapper types
/// for each case (e.g. [`Camera`](struct.Camera.html),
/// [`AmbientLight`](struct.AmbientLight.html),
/// [`Mesh`](struct.Mesh.html), ...).
#[derive(Clone)]
pub struct Object {
    pub(crate) node: NodePointer,
    pub(crate) tx: mpsc::Sender<Message>,
}

impl PartialEq for Object {
    fn eq(
        &self,
        other: &Object,
    ) -> bool {
        self.node == other.node
    }
}

impl Eq for Object {}

impl Hash for Object {
    fn hash<H: Hasher>(
        &self,
        state: &mut H,
    ) {
        self.node.hash(state);
    }
}

impl fmt::Debug for Object {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        self.node.fmt(f)
    }
}

impl Object {
    /// Invisible objects are not rendered by cameras.
    pub fn set_visible(
        &mut self,
        visible: bool,
    ) {
        let msg = Operation::SetVisible(visible);
        let _ = self.tx.send((self.node.downgrade(), msg));
    }

    /// Rotates object in the specific direction of `target`.
    pub fn look_at<E, T>(
        &mut self,
        eye: E,
        target: T,
        up: Option<mint::Vector3<f32>>,
    ) where
        E: Into<mint::Point3<f32>>,
        T: Into<mint::Point3<f32>>,
    {
        use cgmath::{InnerSpace, Point3, Quaternion, Rotation, Vector3};
        let p: [mint::Point3<f32>; 2] = [eye.into(), target.into()];
        let dir = (Point3::from(p[0]) - Point3::from(p[1])).normalize();
        let z = Vector3::unit_z();
        let up = match up {
            Some(v) => Vector3::from(v).normalize(),
            None if dir.dot(z).abs() < 0.99 => z,
            None => Vector3::unit_y(),
        };
        let q = Quaternion::look_at(dir, up).invert();
        self.set_transform(p[0], q, 1.0);
    }

    /// Set both position, orientation and scale.
    pub fn set_transform<P, Q>(
        &mut self,
        pos: P,
        rot: Q,
        scale: f32,
    ) where
        P: Into<mint::Point3<f32>>,
        Q: Into<mint::Quaternion<f32>>,
    {
        let msg = Operation::SetTransform(Some(pos.into()), Some(rot.into()), Some(scale));
        let _ = self.tx.send((self.node.downgrade(), msg));
    }

    /// Add new [`Object`](struct.Object.html) to the group.
    pub fn set_parent<P: AsRef<Object>>(
        &mut self,
        parent: &P,
    ) {
        let msg = Operation::SetParent(parent.as_ref().node.clone());
        let _ = self.tx.send((self.node.downgrade(), msg));
    }

    /// Set position.
    pub fn set_position<P>(
        &mut self,
        pos: P,
    ) where
        P: Into<mint::Point3<f32>>,
    {
        let msg = Operation::SetTransform(Some(pos.into()), None, None);
        let _ = self.tx.send((self.node.downgrade(), msg));
    }

    /// Set orientation.
    pub fn set_orientation<Q>(
        &mut self,
        rot: Q,
    ) where
        Q: Into<mint::Quaternion<f32>>,
    {
        let msg = Operation::SetTransform(None, Some(rot.into()), None);
        let _ = self.tx.send((self.node.downgrade(), msg));
    }

    /// Set scale.
    pub fn set_scale(
        &mut self,
        scale: f32,
    ) {
        let msg = Operation::SetTransform(None, None, Some(scale));
        let _ = self.tx.send((self.node.downgrade(), msg));
    }

    /// Get actual information about itself from the `scene`.
    /// # Panics
    /// Panics if `scene` doesn't have this `Object`.
    pub fn sync(
        &mut self,
        scene: &Scene,
    ) -> Node {
        let mut hub = scene.hub.lock().unwrap();
        hub.process_messages();
        hub.update_graph();
        let node = &hub.nodes[&self.node];
        let root = &hub.nodes[&scene.object.node];
        assert_eq!(node.scene_id, root.scene_id);
        node.into()
    }
}

/// Groups are used to combine several other objects or groups to work with them
/// as with a single entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Group {
    pub(crate) object: Object,
}

impl Group {
    pub(crate) fn new(object: Object) -> Self {
        Group { object }
    }
}
