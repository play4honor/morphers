from abc import ABC, abstractmethod
import polars as pl


BUILTIN_BACKEND_TYPE_MAP = {
    pl.Expr: "polars",
    pl.Series: "polars",
}


class Morpher(ABC):

    BACKEND_LOOKUP = {}

    @classmethod
    def get_backend(cls, backend):
        """
        Gets a backend based on an identifying string, or returns an instantiated backend unchanged.

        - backend: string or MorpherBackend

        Returns: MorpherBackend instance

        If a string is provided, this looks up the correct MorpherBackend subclass
        from the class's BACKEND_LOOKUP dict and _instatiates_ the object.

        This gets called twice on a `from_data` instantiation, the second call will
        always be with an instantiated `MorpherBackend` object.
        """
        try:
            if isinstance(backend, MorpherBackend):
                backend = backend
            else:
                backend = cls.BACKEND_LOOKUP[backend]()
        except KeyError:
            raise ValueError(
                f"'backend' must be a MorpherBackend subclass or one of {list(cls.BACKEND_LOOKUP.keys())}, got {type(backend)}: {backend}"
            )
        return backend

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @classmethod
    def from_data(cls, x, backend=None):
        if backend is None:
            matched_any = False
            for types, backend_name in BUILTIN_BACKEND_TYPE_MAP.items():
                if isinstance(x, types):
                    backend = cls.get_backend(backend_name)
                    matched_any = True
            if not matched_any:
                raise ValueError(
                    f"Provided data of class {type(x)} doesn't match any known backend."
                )
        init_args = backend.from_data(x)
        # The backend here should always be an instantiated morpher backend.
        return cls(**init_args, backend=backend)

    @abstractmethod
    def make_embedding(self):
        raise NotImplementedError

    @abstractmethod
    def make_predictor_head(self):
        raise NotImplementedError

    @abstractmethod
    def make_criterion(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def required_dtype(self):
        raise NotImplementedError

    @abstractmethod
    def save_state_dict(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state_dict):
        raise NotImplementedError


class MorpherBackend(ABC):
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @abstractmethod
    def fill_missing(self, x, missing):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_data(cls, x) -> dict:
        raise NotImplementedError
