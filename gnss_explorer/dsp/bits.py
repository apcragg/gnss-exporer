"""TODO."""


class BitStream:
    """TODO."""

    bitstream: list[int]

    def __init__(self, bitstream: list[int]) -> None:
        """TODO."""
        self.bitstream = bitstream

    def _bitshift(self, bits: list[int], *, twos_complement: bool = False) -> int:
        """Calculate Integer from list of bits where LSBs are first.

        Assumes MSB is first.
        """
        out = 0
        for bit in bits:
            out = (out << 1) | bit

        # Two's complement
        if bits[0] and twos_complement:
            out -= 1 << (len(bits))
        return out

    def _get_field_int(
        self, idx_start: int, n_bits: int, *, scale: int = 1, twos_complement: bool = False
    ) -> int:
        # Bits are specified by 1-index labels from ICD
        idx_start_adj = idx_start - 1
        if idx_start_adj + n_bits >= len(self.bitstream):
            msg = "Index out of range"
            raise ValueError(msg)
        return (
            self._bitshift(
                self.bitstream[idx_start_adj : idx_start_adj + n_bits],
                twos_complement=twos_complement,
            )
            * scale
        )

    def _get_field_int_spanning(
        self,
        fields: list[tuple[int, int]],
        *,
        scale: int = 1,
        twos_complement: bool = False,
    ) -> int:
        bits_concat = []
        for field in fields:
            idx_start_adj = field[0] - 1
            n_bits = field[1]
            if idx_start_adj + n_bits >= len(self.bitstream):
                msg = "Index out of range"
                raise ValueError(msg)
            bits_concat += self.bitstream[idx_start_adj : idx_start_adj + n_bits]
        return (
            self._bitshift(
                bits_concat,
                twos_complement=twos_complement,
            )
            * scale
        )

    def _get_field_real(
        self, idx_start: int, n_bits: int, scale: float, *, twos_complement: bool = False
    ) -> float:
        return self._get_field_int(idx_start, n_bits, twos_complement=twos_complement) * scale

    def _get_field_real_spanning(
        self, fields: list[tuple[int, int]], scale: float, *, twos_complement: bool = False
    ) -> float:
        return self._get_field_int_spanning(fields, twos_complement=twos_complement) * scale
