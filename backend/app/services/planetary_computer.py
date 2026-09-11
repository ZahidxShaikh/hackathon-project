"""Connection helpers for Microsoft Planetary Computer STAC."""

from pystac_client import Client
import planetary_computer

from backend.app.config import PLANETARY_COMPUTER_STAC_URL


class PlanetaryComputerConnectionError(RuntimeError):
    """Raised when the Planetary Computer STAC catalog cannot be opened."""


def open_planetary_computer_catalog() -> Client:
    """Open Planetary Computer with its official asset-signing modifier.

    The modifier signs asset URLs should a later phase need to read a selected
    asset. Phase 2 only reads STAC metadata and does not download imagery.
    """
    try:
        return Client.open(
            PLANETARY_COMPUTER_STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )
    except Exception as error:  # pystac-client may raise requests/STAC errors.
        raise PlanetaryComputerConnectionError(
            "Could not connect to Microsoft Planetary Computer STAC. "
            "Check your network connection and try again."
        ) from error

