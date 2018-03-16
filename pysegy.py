import numpy as np

segy_file = '../input/F3_Demo_2016_training_v6/Rawdata/Seismic_data.sgy'

# Some definitions
SEGY_BINARY_HEADER_SIZE = 400
SEGY_TEXT_HEADER_SIZE = 3200
SEGY_TRACE_HEADER_SIZE = 240

def np_traces(fname, trace0, n_samples, sample_format):
    segy_trace_dtype = np.dtype([
        ('header', (bytes, SEGY_TRACE_HEADER_SIZE)),
        ('trace',  (sample_format, n_samples))
    ])
    
    segy_raw = np.memmap(fname, dtype=segy_trace_dtype, offset=trace0, mode='r')
    
    return segy_raw

trace0 = SEGY_BINARY_HEADER_SIZE+SEGY_TEXT_HEADER_SIZE
n_samples = 462
sample_format = '>i2'

segy_raw = np_traces(segy_file, trace0, n_samples, sample_format)
all_traces = segy_raw['trace']
a_trace = all_traces[10000]

print(segy_raw.shape)   # outputs (600515,)
print(all_traces.shape) # outputs (600515, 462)
print(a_trace.shape)    # outputs (462,)

Performance-wise, it seems to be very fast:

%time segy_raw['trace'].max()
# CPU times: user 456 ms, sys: 70.3 ms, total: 526 ms
# Wall time: 523 ms
# 32767

%time segy_raw['trace'].min()
# CPU times: user 442 ms, sys: 3.94 ms, total: 446 ms
# Wall time: 443 ms
# -32767 
