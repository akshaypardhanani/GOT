"""GOT-JEPA tracker entry point (ToMPnet_PT / CoTracker based)."""

from ..tomp.tomp_gotejepa import ToMP


def get_tracker_class():
    return ToMP
