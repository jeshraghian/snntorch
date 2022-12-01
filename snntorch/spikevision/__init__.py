from warnings import warn

warn(
    f"The module {__name__} is deprecated. For loading "
    f"neuromorphic datasets, we recommend using the Tonic project: "
    f"https://github.com/neuromorphs/tonic",
    DeprecationWarning,
    stacklevel=2,
)
