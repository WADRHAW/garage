"""Base Garage Environment API."""

import abc


class Environment(abc.ABC):
    """The main API for garage environments.

    The public API methods are:

    +-----------------------+
    | Functions             |
    +=======================+
    | reset()               |
    +-----------------------+
    | step()                |
    +-----------------------+
    | render()              |
    +-----------------------+
    | visualize()           |
    +-----------------------+
    | close()               |
    +-----------------------+

    Set the following properties:

    +-----------------------+-------------------------------------------------+
    | Properties            | Description                                     |
    +=======================+=================================================+
    | action_space          | The action space specification                  |
    +-----------------------+-------------------------------------------------+
    | observation_space     | The observation space specification             |
    +-----------------------+-------------------------------------------------+
    | spec                  | The environment specifications                  |
    +-----------------------+-------------------------------------------------+
    | render_modes          | The list of supported render modes              |
    +-----------------------+-------------------------------------------------+

    Example of a simple rollout loop:

    .. code-block:: python

        env = MyEnv()
        policy = MyPolicy()
        first_ts = env.reset()
        env.visualize()  # visualization window opened

        episode = [first_ts, ]
        while not ts.last():
            action = policy.get_action(episode[-1].next_observation)
            episode.append(env.step(action))

        env.close()  # visualization window closed

    Make sure your environment is pickle-able:
        Garage pickles the environment via the `cloudpickle` module
        to save snapshots of the experiment. However, some environments may
        contain attributes that are not pickle-able (e.g. a client-server
        connection). In such cases, override `__setstate__()` and
        `__getstate()__` to add your custom pickle logic.

        You might want to refer to the EzPickle module:
        https://github.com/openai/gym/blob/master/gym/utils/ezpickle.py
        for a lightweight way of pickle and unpickle via constructor
        arguments.

    """

    @property
    @abc.abstractmethod
    def action_space(self):
        """akro.Space: The action space specification."""

    @property
    @abc.abstractmethod
    def observation_space(self):
        """akro.Space: The observation space specification."""

    @property
    @abc.abstractmethod
    def spec(self):
        """EnvSpec: The environment specification."""

    @property
    @abc.abstractmethod
    def render_modes(self):
        """list: A list of string representing the supported render modes.

        See render() for a list of modes.
        """

    @abc.abstractmethod
    def reset(self):
        """Resets the environment and returns the first `TimeStep` of sequence.

        Returns:
            TimeStep: The first time step. Note that this `TimeStep`
            will have attributes `observation`, `action`, `reward` as None
            as they are undefined.

        """

    @abc.abstractmethod
    def step(self, action):
        """Steps the environment using to the action and returns a `TimeStep`.

        If the environment returned the last time step of a sequence (either
        because of terminal or time out) at the previous step, this call to
        `step()` will start a new sequence and `action` will be ignored.

        This method will also start a new sequence if called after the
        environment has been constructed and `reset()` has not been called.
        Again, in this case `action` will be ignored.

        If `spec.max_episode_steps` is reached after applying the action
        and the sequence is not done, `step()` should return a `TimeStep` with
        `step_type==StepType.TIMEOUT`.

        If possible, update the visualization display as well.

        Args:
            action (np.ndarray): A NumPy array, or a nested dict, list or tuple
                of arrays corresponding to `action_space`.

        Returns:
            TimeStep: The time step after the action.

        """

    @abc.abstractmethod
    def render(self, mode):
        """Renders the environment.

        The set of supported modes varies per environment. By convention,
        if mode is:

        * rgb_array: Return an `numpy.ndarray` with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        * ansi: Return a string (str) or `StringIO.StringIO` containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Make sure that your class's `render_modes` includes the list of
        supported modes.

        For example:

        .. code-block:: python

            class MyEnv(Environment):
                def render_modes(self):
                    return ['rgb_array', 'ansi']

                def render(self, mode):
                    if mode == 'rgb_array':
                        return np.array(...)  # return RGB frame for video
                    elif mode == 'ansi':
                        ...  # return text output
                    else:
                        raise ValueError('Supported render modes are {}, but '
                                         'got render mode {} instead.'.format(
                                             self.render_modes, mode))

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        """

    @abc.abstractmethod
    def visualize(self):
        """Creates a visualization of the environment.

        This function should be called **only once** after `reset()` to set up
        the visualization display. The visualization should be updated
        when the environment is changed (i.e. when `step()` is called.)

        Calling `close()` will deallocate any resources and close any
        windows created by `visualize()`. If `close()` is not explicitly
        called, the visualization will be closed when the environment is
        destructed (i.e. garbage collected).

        """

    @abc.abstractmethod
    def close(self):
        """Closes the environment.

        This method should close all windows invoked by `visualize()`.

        Override this function in your subclass to perform any necessary
        cleanup.

        Environments will automatically `close()` themselves when they are
        garbage collected or when the program exits.
        """

    def __del__(self):
        """Environment destructor."""
        self.close()
