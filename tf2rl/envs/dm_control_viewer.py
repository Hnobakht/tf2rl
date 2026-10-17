"""
https://github.com/rlworkgroup/garage.git
Wrapper for the dm_control viewer which allows single-stepping
"""
import dm_control.viewer.application as dm_viewer_app
import glfw
import time

class DmControlViewer(dm_viewer_app.Application):
    """dm_control viewer exposing rendering and single-step control to the caller."""
    def render(self):
        # Don't try to render into closed windows
        """Render the viewer and handle events, continuing to redraw while paused."""
        if not self._window._context:
            return

        self._render_once()
        if self._time_multiplier.get() < 1:
            time.sleep(0.01/self._time_multiplier.get())

        # Just keep rendering if we're paused, but hold onto control flow
        while self._pause_subject.value:
            self._render_once()

    def _render_once(self):
        # See https://github.com/deepmind/dm_control/blob/92f9913013face0468442cd0964d5973ea2089ea/dm_control/viewer/gui/glfw_gui.py#L280  # noqa: E501
        """Render one viewer frame and process input events, closing the window if needed."""
        window = self._window
        tick_func = self._tick_func
        if (window._context
                and not glfw.window_should_close(window._context.window)):
            pixels = tick_func()
            with window._context.make_current() as ctx:
                ctx.call(window._update_gui_on_render_thread,
                         window._context.window, pixels)
            window._mouse.process_events()
            window._keyboard.process_events()
        else:
            window.close()

    def launch(self, environment_loader, policy=None):
        # See https://github.com/deepmind/dm_control/blob/92f9913013face0468442cd0964d5973ea2089ea/dm_control/viewer/application.py#L304  # noqa: E501
        """Load the environment and optional policy, leaving the viewer ready for single stepping.
        """
        if environment_loader is None:
            raise ValueError('"environment_loader" argument is required.')
        if callable(environment_loader):
            self._environment_loader = environment_loader
        else:
            self._environment_loader = lambda: environment_loader
        self._policy = policy
        self._load_environment(zoom_to_scene=True)

        def tick():
            """Advance the viewer and return the rendered pixels."""
            self._viewport.set_size(*self._window.shape)
            self._tick()
            return self._renderer.pixels

        self._tick_func = tick

        # Start unpaused
        self._pause_subject.value = False

    def close(self):
        """Close the viewer window."""
        self._window.close()

    def restart(self):
        """Restart the viewer runtime for the current environment."""
        self._restart_runtime()
