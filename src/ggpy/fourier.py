import numpy as np
import matplotlib.pyplot as plt


def ft(signal, f_shift, x_shift):
    """
    This functions computes the fourier transform of the given signal.
    For signals of even length, the zero frequency component is ignored.
    This equates to it being assumed to be zero. The frequencies are centers at
    (N - 1) / 2. Each frequency is calculated from that point.

    For the purposes of this exercises, the signal length is assumed to be even.
    Even better, it should be a power of 2. (To implement fft).

    :param signal: array: Shape (N,) The signal to be fourier transformed
    :param f_shift: integral number
    :param x_shift: integral number

    :return: transform: the frequency space representation of the signal
        shape: (N, )
        Note: for a pure real signal, the real part of the transform is even,
              while the imaginary part is odd. (across the midpoint).
    """

    N = signal.shape[0]

    # Make the position and frequency arrays. Shape matters for broadcasting
    position = np.arange(N)
    frequency = np.arange(N).reshape(N, 1)

    # Apply the DFT, note that the frequency and position indices are the same
    transform = (
        np.exp(-2j * np.pi * (frequency - f_shift) * (position - x_shift) / N) @ signal
    )

    return transform


def ift(transform, f_shift, x_shift):
    """
    The inverse of the function ft.
    Computes the position/time space signal based on the frequency input.

    :param signal:
    :return:
    """

    N = transform.shape[0]

    # Have to change the indices to frequency components. (unitless)
    positions = np.arange(N)
    frequencies = np.arange(N).reshape(N, 1)

    # Compute it
    signal = transform @ np.exp(
        2j * np.pi * (positions - x_shift) * (frequencies - f_shift) / N
    )

    # Need to normalize the signal (using cosmology convention).
    return signal / N


def fft(signal, f_shift, x_shift):
    """
    This functions is the fast version of the ft function.
    It utilizes the Cooley and Tukey fast fourier transform algorithm.
    It was modified to work with the shifted DFT.

    :param signal: The signal to be transformed
    :return: transform: The transformed signal.
    """

    N = signal.shape[0]

    # Need to convert the index of the array to the position correspondence
    k = np.arange(N)

    # Check the size of the array to determine if symmetry should be applied or not
    if N <= 32:
        return ft(signal, f_shift, x_shift)
    else:
        even = fft(signal[::2], f_shift, x_shift / 2)
        odd = fft(signal[1::2], f_shift, x_shift / 2)

        alpha = np.exp(1j * np.pi * x_shift)
        beta = np.exp(-2j * np.pi * (k - f_shift) / N)

        return np.concatenate(
            [even + beta[: N // 2] * odd, alpha * (even + beta[N // 2 :] * odd)]
        )


def fft_v(signal):
    """
    This functions is the fast version of the ft function.
    It utilizes the Cooley and Tukey fast fourier transform algorithm.
    It was modified to work with the shifted DFT.
    Also was vectorized for increased efficiency.

    :param signal: The signal to be transformed
    :return: transform: The transformed signal.
    """

    N = signal.shape[0]
    shift = (N - 1) / N  # This only works for the centered DFT
    f_shift = (N - 1) / 2  # The f_shift is constant through-out
    N_min = min(N, 2)

    # Need to convert the index of the array to the position correspondence
    x = np.arange(N_min)
    k = x[:, None]

    term = np.exp(-2j * np.pi * (k - f_shift) * (x - shift) / N_min)
    X = term @ signal.reshape((N_min, -1))

    # Now build the array up
    while X.shape[0] < N:
        shift *= 2

        even = X[:, : X.shape[1] // 2]
        odd = X[:, X.shape[1] // 2 :]

        beta = np.exp(-1j * np.pi * (np.arange(X.shape[0]) - f_shift) / X.shape[0])[
            :, None
        ]
        alpha = np.exp(1j * np.pi * shift)

        X = np.vstack(
            [even + beta * odd, alpha * (even - beta * odd)]
        )  # There's a bug yay

    return X.ravel()  # Now to check if it works xD


def ifft(transform, f_shift, x_shift):
    """
    Fast version of the inverse fourier transform
    Designed to be the inverse of fft, and works on the same principles.
    """

    N = transform.shape[0]

    # Need to convert the index of the array to the position correspondence
    x = np.arange(N)

    # Check the size of the array to determine if symmetry should be applied or not
    if N <= 32:
        return ift(transform, f_shift, x_shift)
    else:
        even = ifft(transform[::2], f_shift / 2, x_shift)
        odd = ifft(transform[1::2], f_shift / 2, x_shift)

        alpha = np.exp(-1j * np.pi * f_shift)
        beta = np.exp(2j * np.pi * (x - x_shift) / N)

        return (
            np.concatenate(
                [even + beta[: N // 2] * odd, alpha * (even + beta[N // 2 :] * odd)]
            )
            / 2
        )


def ifft_v(signal):
    """
    This functions is the fast version of the ft function.
    It utilizes the Cooley and Tukey fast fourier transform algorithm.
    It was modified to work with the shifted DFT.
    Also was vectorized for increased efficiency.

    :param signal: The signal to be transformed
    :return: transform: The transformed signal.
    """

    N = signal.shape[0]
    shift = (N - 1) / N  # This only works for the centered DFT
    x_shift = (N - 1) / 2  # The f_shift is constant through-out
    N_min = min(N, 2)

    # Need to convert the index of the array to the position correspondence
    k = np.arange(N_min)
    x = k[:, None]

    term = np.exp(2j * np.pi * (x - x_shift) * (k - shift) / N_min)
    X = term @ signal.reshape((N_min, -1))

    # Now build the array up
    while X.shape[0] < N:
        shift *= 2

        even = X[:, : X.shape[1] // 2]
        odd = X[:, X.shape[1] // 2 :]

        beta = np.exp(1j * np.pi * (np.arange(X.shape[0]) - x_shift) / X.shape[0])[
            :, None
        ]
        alpha = np.exp(-1j * np.pi * shift)

        X = np.vstack(
            [even + beta * odd, alpha * (even - beta * odd)]
        )  # There's a bug yay

    # Normalization removed for the purposes of this project
    return X.ravel()  # Now to check if it works xD


def ft2(signal, f_shift, x_shift):
    """
    Brute force version of the two-dimensional fourier transform.

    Ideally, the shape of the signal is of the form (2^n, 2^n) for some integer n.
    This makes conversion to fft easier, and assures we can take uniform shifts for a centered DFT.

    :param signal: shape (M, N) the signal to be transformed / decomposed into fourier components
    :param f_shift: shape (2,) the frequency shift on the x and y axes respectively
    :param x_shift: shape (2,) the position shift on x and y respectively
    :return: transform: shape(M, N) the fourier representation of the signal.
    """
    M, N = signal.shape[0], signal.shape[1]
    m, n = np.arange(M), np.arange(N)
    m, n = np.meshgrid(m, n)

    transform = np.zeros((M, N), dtype=np.complex128)

    for u in np.arange(M):
        for v in np.arange(N):
            transform[u, v] = np.sum(
                signal
                * np.exp(
                    -2j
                    * np.pi
                    * (
                        (u - f_shift[0]) * (m - x_shift[0]) / M
                        + (v - f_shift[1]) * (n - x_shift[1]) / N
                    )
                )
            )

    return transform


def ift2(transform, f_shift, x_shift):
    """
    Brute force version of the inverse two-dimensional fourier transform.

    :param transform: shape (M, N) the signal to be transformed / decomposed into fourier components
    :param f_shift: shape (2,) the frequency shift on the x and y axes respectively
    :param x_shift: shape (2,) the position shift on x and y respectively
    :return: signal: shape(M, N) the fourier representation of the signal.
    """
    M, N = transform.shape[0], transform.shape[1]
    m, n = np.arange(M), np.arange(N)
    u, v = np.meshgrid(m, n)

    signal = np.zeros((M, N), dtype=np.complex128)

    for x in np.arange(M):
        for y in np.arange(N):
            signal[y, x] = np.sum(
                transform
                * np.exp(
                    2j
                    * np.pi
                    * (
                        (u - f_shift[0]) * (x - x_shift[0]) / M
                        + (v - f_shift[1]) * (y - x_shift[1]) / N
                    )
                )
            )

    # Needs to be normalized
    return signal / M / N


def fft2(signal, f_shift, x_shift):
    """
    This functions is the fast version of the ft2 function.
    It utilizes the Cooley and Tukey fast fourier transform algorithm.
    It was modified to work with the shifted DFT, in 2-D.
    It just calculates the fft column-wise, and then row-wise of the modified signal.

    :param signal: The signal to be transformed
    The idea is the same as all the other transforms
    :return: transform: The transformed signal.
    """

    M, N = signal.shape

    # Perform the column-wise pass
    first_pass = np.zeros((M, N), dtype=np.complex128)

    for column in range(N):
        first_pass[:, column] = fft(signal[:, column], f_shift[1], x_shift[1])
        # Have to check the shifting - is v confusing
        # Anyway for my use case it's symmetric

    # Now the row-wise pass
    out = np.zeros((M, N), dtype=np.complex128)

    for row in range(M):
        out[row] = fft(first_pass[row], f_shift[0], x_shift[0])

    return out


def fft2_v(signal):
    """
    This functions is the fast version of the ft2 function.
    It utilizes the Cooley and Tukey fast fourier transform algorithm.
    It was modified to work with the shifted DFT, in 2-D.
    It just calculates the fft column-wise, and then row-wise of the modified signal.

    :param signal: The signal to be transformed
    The idea is the same as all the other transforms
    :return: transform: The transformed signal.
    """

    M, N = signal.shape

    # Perform the column-wise pass
    first_pass = np.zeros((M, N), dtype=np.complex128)

    for column in range(N):
        first_pass[:, column] = fft_v(signal[:, column])
        # Have to check the shifting - is v confusing
        # Anyway for my use case it's symmetric

    # Now the row-wise pass
    out = np.zeros((M, N), dtype=np.complex128)

    for row in range(M):
        out[row] = fft_v(first_pass[row])

    return out


def ifft2(transform, f_shift, x_shift):
    """
    Fast version of the 2-D inverse fourier transform
    Designed to be the inverse of fft2, and works on the same principles.
    """

    M, N = transform.shape

    # Perform the row-wise pass
    first_pass = np.zeros((M, N), dtype=np.complex128)

    for row in range(M):
        first_pass[row] = ifft(transform[row], f_shift[0], x_shift[0])

    # Now the column-wise pass
    out = np.zeros((M, N), dtype=np.complex128)

    for column in range(N):
        out[:, column] = ifft(first_pass[:, column], f_shift[1], x_shift[1])

    # No need to normalize, ifft takes care of it
    return out


def ifft2_v(transform):
    """
    Fast version of the 2-D inverse fourier transform
    Designed to be the inverse of fft2, and works on the same principles.
    """

    M, N = transform.shape

    # Perform the row-wise pass
    first_pass = np.zeros((M, N), dtype=np.complex128)

    for row in range(M):
        first_pass[row] = ifft_v(transform[row])

    # Now the column-wise pass
    out = np.zeros((M, N), dtype=np.complex128)

    for column in range(N):
        out[:, column] = ifft_v(first_pass[:, column])

    # No need to normalize, ifft takes care of it
    return out


if __name__ == "__main__":
    N = 32
    space = np.linspace(0, 2 * np.pi, N * N).reshape(N, N)
    signal = space  # np.cos(space) ** 2 * np.sin(space) ** 2
    shift = np.array([(N - 1) / 2, (N - 1) / 2])

    plt.plot(signal)
    plt.show()

    true = ifft2(signal, shift, shift)
    test = ifft2_v(signal)

    plt.imshow(np.real(true))
    plt.imshow(np.imag(true))
    plt.show()

    plt.imshow(np.real(test))
    plt.imshow(np.imag(test))
    plt.show()
    print(np.allclose(true, test))

    # k = 25
    # slow = []
    # for i in range(k):
    #    start = time.time()
    #    fft2(signal, shift, shift)
    #    slow.append(time.time() - start)
    #
    # print(np.asarray(slow).mean())
    #
    # fast = []
    # for i in range(k):
    #    start = time.time()
    #    fft2_v(signal)
    #    fast.append(time.time() - start)
    #
    # print(np.asarray(fast).mean())

    # Right now, it all seems to work properly
    """
    Correctness of the following functions has been established.
    1. ft     and ift
    2. fft    and ifft 
    3. ft2    and ift2
    4. fft2   and ifft2
    5. fft_v  and ifft_v
    6. fft2_v and ifft2_v
    """
