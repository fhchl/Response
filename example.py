from response import Response
fs = 16
x = [1]
# Create response and do chain of processing
r = Response.from_time(fs, x) \
            .zeropad(0, 15) \
            .delay(0.5) \
            .resample(10 * fs, window=('kaiser', 0.5)) \
            .timecrop(0, 0.6) \
            .time_window((0, 0.2), (0.5, 0.6))
# plot result
r.plot(show=True)
# time domain data
r.in_time
# frequency domain data
r.in_freq
