from collections.abc import Callable
from typing import Optional

import jax
from jax.experimental import host_callback
from tqdm.auto import tqdm


def fori_loop_tqdm_decorator(
    n_iterations: int,
    print_every: Optional[int] = None,
    **kwargs,
) -> Callable:
    """Add a tqdm progress bar to `body_fun` used in `jax.lax.fori_loop`.
    This function is based on the implementation in [`jax_tqdm`](https://github.com/jeremiecoullon/jax-tqdm)
    """

    _update_progress_bar, close_tqdm = _build_tqdm(n_iterations, print_every, **kwargs)

    def _fori_loop_tqdm_decorator(func):
        def wrapper_progress_bar(i, val):
            _update_progress_bar(i)
            result = func(i, val)
            return close_tqdm(result, i)

        return wrapper_progress_bar

    return _fori_loop_tqdm_decorator


def _build_tqdm(
    n_iterations: int,
    print_every: Optional[int] = None,
    **kwargs,
) -> tuple[Callable, Callable]:
    """Build the tqdm progress bar on the host."""

    desc = kwargs.pop("desc", f"Running for {n_iterations:,} iterations")
    message = kwargs.pop("message", desc)
    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    tqdm_bars = {}

    if print_every is None:
        if n_iterations > 20:
            print_every = int(n_iterations / 20)
        else:
            print_every = 1
    else:
        if print_every < 1:
            raise ValueError(
                "The number of iterations per progress bar update should "
                f"be greater than 0. Got {print_every}."
            )
        elif print_every > n_iterations:
            raise ValueError(
                "The number of iterations per progress bar update should be less "
                f"than the number of iterations, equal to {n_iterations}. "
                f"Got {print_every}."
            )

    remainder = n_iterations % print_every

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(n_iterations), **kwargs)
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_every == 0) & (iter_num != n_iterations - remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_every, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == n_iterations - remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return jax.lax.cond(
            iter_num == n_iterations - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    return _update_progress_bar, close_tqdm
