"""SFQ - Super Fast and Quality: Solar vector magnetogram azimuth disambiguation.

Reference: Rudenko, G. V. & Anfinogentov, S. A. (2014),
"Very Fast and Accurate Azimuth Disambiguation of Vector Magnetograms",
Solar Physics, Volume 289, Issue 5, pp. 1499-1516.

This tree includes IDL-parity fixes for geometry-aware ``pot_vmag``,
``sfq_clean`` filter width, and image-axis layout (see changelog / PR).
"""

from sfq.disambig import sfq_disambig

__all__ = ["sfq_disambig"]
