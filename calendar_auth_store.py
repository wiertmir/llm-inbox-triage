"""Native credential storage shared by calendar OAuth providers."""

import sys

from keyring.backend import KeyringBackend


def get_credential_store() -> KeyringBackend:
    if sys.platform == "win32":
        from keyring.backends.Windows import WinVaultKeyring

        return WinVaultKeyring()
    if sys.platform == "darwin":
        from keyring.backends.macOS import Keyring

        return Keyring()
    from keyring.backends.SecretService import Keyring

    return Keyring()
