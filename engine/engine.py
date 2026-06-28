import ctypes
import os

class ChessEngine:
    def __init__(
        self,
        dll_path,
        model_path,
        tt_size=2000000,
        eval_cache_size=524288,
        emit_search_info=False,
    ):
        # Load the newly compiled unified shared library
        self.lib = ctypes.CDLL(os.path.abspath(dll_path))

        # Configure Argument Types for the Interface Handshake
        self.lib.load_nnue_model_and_caches.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.load_nnue_model_and_caches.restype = ctypes.c_bool

        self.lib.load_fen_to_native_search.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_char_p]
        self.lib.load_fen_to_native_search.restype = ctypes.c_bool
        self.lib.search_position_native.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        self.lib.search_position_native.restype = ctypes.c_bool
        self._score_api_available = hasattr(self.lib, "get_last_search_score_native")
        if self._score_api_available:
            self.lib.get_last_search_score_native.argtypes = []
            self.lib.get_last_search_score_native.restype = ctypes.c_int
        self._quantized_api_available = hasattr(self.lib, "set_quantized_inference_native")
        if self._quantized_api_available:
            self.lib.set_quantized_inference_native.argtypes = [ctypes.c_bool]
            self.lib.set_quantized_inference_native.restype = None
        self._search_info_api_available = hasattr(self.lib, "set_native_search_info_enabled")
        if self._search_info_api_available:
            self.lib.set_native_search_info_enabled.argtypes = [ctypes.c_bool]
            self.lib.set_native_search_info_enabled.restype = None

        # Initialize NNUE weights and power-of-two memory allocations
        model_bytes = model_path.encode('utf-8')
        success = self.lib.load_nnue_model_and_caches(model_bytes, tt_size, eval_cache_size)
        if not success:
            raise RuntimeError("Engine initialization failed! Verify NNUE file path.")
        self.set_search_info_enabled(emit_search_info)

    def get_best_move(self, fen: str, depth: int, time_limit_ms: float) -> str:
        # Prepare a mutable output buffer string for UCI moves (e.g. "e2e4\0")
        output_buffer = ctypes.create_string_buffer(8)

        # Call into our native engine shell block
        success = self.lib.load_fen_to_native_search(
            fen.encode('utf-8'),
            ctypes.c_int(depth),
            ctypes.c_double(time_limit_ms),
            output_buffer
        )
        if not success:
            raise RuntimeError("Native search failed. Check the FEN and loaded NNUE model.")

        return output_buffer.value.decode('utf-8')

    def get_best_move_with_score(self, fen: str, depth: int, time_limit_ms: float) -> tuple[str, int]:
        """Return the best UCI move and root centipawn score from the latest native search."""
        best_move = self.get_best_move(fen, depth, time_limit_ms)
        score = self.get_last_search_score()
        return best_move, score

    def get_best_move_with_clock(
        self,
        fen: str,
        depth: int,
        wtime: int,
        btime: int,
        winc: int = 0,
        binc: int = 0,
        movestogo: int = 0,
    ) -> str:
        output_buffer = ctypes.create_string_buffer(8)
        success = self.lib.search_position_native(
            fen.encode('utf-8'),
            ctypes.c_int(depth),
            ctypes.c_int(wtime),
            ctypes.c_int(btime),
            ctypes.c_int(winc),
            ctypes.c_int(binc),
            ctypes.c_int(movestogo),
            output_buffer,
        )
        if not success:
            raise RuntimeError("Native search failed. Check the FEN and loaded NNUE model.")

        return output_buffer.value.decode('utf-8')

    def get_last_search_score(self) -> int:
        if not self._score_api_available:
            return 0
        return int(self.lib.get_last_search_score_native())

    def clear_caches(self) -> None:
        if hasattr(self.lib, "clear_native_eval_cache"):
            self.lib.clear_native_eval_cache()

    def set_quantized_inference(self, enabled: bool) -> None:
        if self._quantized_api_available:
            self.lib.set_quantized_inference_native(ctypes.c_bool(enabled))

    def set_search_info_enabled(self, enabled: bool) -> None:
        if self._search_info_api_available:
            self.lib.set_native_search_info_enabled(ctypes.c_bool(enabled))
