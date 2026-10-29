"""
model_builder.py
A small library for programatically building MuJoCo XML files
"""
from contextlib import contextmanager
import tempfile
import numpy as np


def default_model(name):
    """Get a model with basic settings such as gravity and RK4 integration enabled

    Args:
        name (str): MJCF model name or XML element tag to create.

    Returns:
        MJCModel: MJCF model builder initialized with common defaults, compiler, physics, and visual
            settings.
    """
    model = MJCModel(name)
    root = model.root

    # Setup
    root.compiler(angle="radian", inertiafromgeom="true")
    default = root.default()
    default.joint(armature=1, damping=1, limited="true")
    default.geom(contype=0, friction='1 0.1 0.1', rgba='0.7 0.7 0 1')
    root.option(gravity="0 0 -9.81", integrator="RK4", timestep=0.01)
    return model

def pointmass_model(name):
    """Get a model with basic settings such as gravity and Euler integration enabled

    Args:
        name (str): MJCF model name or XML element tag to create.

    Returns:
        MJCModel: MJCF model builder initialized with point-mass defaults and physics settings.
    """
    model = MJCModel(name)
    root = model.root

    # Setup
    root.compiler(angle="radian", inertiafromgeom="true", coordinate="local")
    default = root.default()
    default.joint(limited="false", damping=1)
    default.geom(contype=2, conaffinity="1", condim="1", friction=".5 .1 .1", density="1000", margin="0.002")
    root.option(timestep=0.01, gravity="0 0 0", iterations="20", integrator="Euler")
    return model


class MJCModel(object):
    """MuJoCo XML model with helpers for tree construction and temporary-file export."""
    def __init__(self, name):
        """Create a named MuJoCo XML model with an empty root element.

        Args:
            name (str): MJCF model name or XML element tag to create.

        Returns:
            None: Initializes the instance.
        """
        self.name = name
        self.root = MJCTreeNode("mujoco").add_attr('model', name)

    @contextmanager
    def asfile(self):
        """Usage:
        model = MJCModel('reacher')
        with model.asfile() as f:
            print f.read()  # prints a dump of the model

        Yields:
            file-like object: Open temporary MJCF text file, valid only inside the context-manager
                block; closed and deleted on exit.
        """
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

    def open(self):
        """Write the model XML to a temporary file and return its rewound handle.

        Returns:
            file-like object: Open temporary MJCF text file assigned to self.file; call close() to close
                and delete it.
        """
        self.file = tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=True)
        self.root.write(self.file)
        self.file.seek(0)
        return self.file

    def close(self):
        """Close and delete the temporary XML file opened for this model.

        Returns:
            None.
        """
        self.file.close()

    def find_attr(self, attr, value):
        """Find the first XML node with the requested attribute value.

        Args:
            attr (str): XML attribute name to search.
            value (object): XML attribute value to store or compare with the requested attribute.

        Returns:
            MJCTreeNode or None: First node with the matching attribute/value in depth-first order.
        """
        return self.root.find_attr(attr, value)

    def __getstate__(self):
        """Return the empty serialized state used by this legacy model.

        Returns:
            dict: Empty serialization state; the historical implementation intentionally excludes model
                attributes.
        """
        return {}

    def __setstate__(self, state):
        """Accept serialized state without restoring any model attributes.

        Args:
            state (dict): Deserialized state, ignored by this historical no-op restoration hook.

        Returns:
            None.
        """
        pass


class MJCTreeNode(object):
    """XML tree node supporting chained attributes and dynamic child-tag creation."""
    def __init__(self, name):
        """Create an XML element with no attributes or child nodes.

        Args:
            name (str): MJCF model name or XML element tag to create.

        Returns:
            None: Initializes the instance.
        """
        self.name = name
        self.attrs = {}
        self.children = []

    def add_attr(self, key, value):
        """Set an XML attribute, converting numeric sequences to space-separated text.

        Args:
            key (str): XML attribute name to set.
            value (object): XML attribute value to store or compare with the requested attribute.

        Returns:
            MJCTreeNode: This node after storing the attribute, permitting chained builder calls.
        """
        if isinstance(value, str):
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val).lower() for val in value])
        else:
            value = str(value).lower()

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        """Return a factory that appends an XML child with the requested tag.

        Args:
            name (str): MJCF model name or XML element tag to create.

        Returns:
            callable: Factory accepting XML attribute keywords, appending a child with the requested
                tag, and returning it.
        """
        def wrapper(**kwargs):
            """Create and append an XML child node with the supplied attributes.

            Args:
                **kwargs (dict): XML attribute names mapped to values for the appended child element.

            Returns:
                MJCTreeNode: Newly appended child carrying the supplied XML attributes.
            """
            newnode =  MJCTreeNode(name)
            for (k, v) in kwargs.items():
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode
        return wrapper

    def dfs(self):
        """Yield this XML node and its descendants in depth-first order.

        Yields:
            MJCTreeNode: This node followed by descendants in depth-first order.
        """
        yield self
        if self.children:
            for child in self.children:
                for node in child.dfs():
                    yield node

    def find_attr(self, attr, value):
        """Run DFS to find a matching attr

        Args:
            attr (str): XML attribute name to search.
            value (object): XML attribute value to store or compare with the requested attribute.

        Returns:
            MJCTreeNode or None: First depth-first match for the attribute/value pair, or None if
                absent.
        """
        if attr in self.attrs and self.attrs[attr] == value:
            return self
        for child in self.children:
            res = child.find_attr(attr, value)
            if res is not None:
                return res
        return None


    def write(self, ostream, tabs=0):
        """Write this XML subtree to a stream with indentation.

        Args:
            ostream (file-like object): Writable text stream receiving the serialized XML subtree.
            tabs (int): Initial indentation depth in tab characters.

        Returns:
            None.
        """
        contents = ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        if self.children:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s>\n' % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs+1)
            ostream.write('\t'*tabs)
            ostream.write('</%s>\n' % self.name)
        else:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s/>\n' % (self.name, contents))

    def __str__(self):
        """Return the opening XML tag and its attributes.

        Returns:
            str: Opening XML tag with this node attributes, without serializing children or a closing
                tag.
        """
        s = "<"+self.name
        s += ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        return s+">"
