�	� �r��@� �r��@!� �r��@	Ya�i��?Ya�i��?!Ya�i��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$� �r��@�蹅�D�?A��)@Yh��|?u)@*	    ���@2f
/Iterator::Model::MaxIntraOpParallelism::BatchV2����Mb)@!u�Bw�X@)�����Y)@16�2��X@:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle �� �rh�?!W~T �!�?)�� �rh�?1W~T �!�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismL7�A`e)@!|��}�X@)�~j�t�x?1�9h��/�?:Preprocessing2F
Iterator::Model���x�f)@!      Y@)�~j�t�h?1�9h��/�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Ya�i��?I�{Xz�X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�蹅�D�?�蹅�D�?!�蹅�D�?      ��!       "      ��!       *      ��!       2	��)@��)@!��)@:      ��!       B      ��!       J	h��|?u)@h��|?u)@!h��|?u)@R      ��!       Z	h��|?u)@h��|?u)@!h��|?u)@b      ��!       JCPU_ONLYYYa�i��?b q�{Xz�X@Y      Y@q�>(u��r?"�
device�Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 