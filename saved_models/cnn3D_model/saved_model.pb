��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18��
�
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:*
dtype0
w
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ϊ�*
shared_namedense/kernel
p
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*!
_output_shapes
:Ϊ�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
RMSprop/conv3d/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/conv3d/kernel/rms
�
-RMSprop/conv3d/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3d/kernel/rms**
_output_shapes
:*
dtype0
�
RMSprop/conv3d/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameRMSprop/conv3d/bias/rms

+RMSprop/conv3d/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3d/bias/rms*
_output_shapes
:*
dtype0
�
RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ϊ�*)
shared_nameRMSprop/dense/kernel/rms
�
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*!
_output_shapes
:Ϊ�*
dtype0
�
RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameRMSprop/dense/bias/rms
~
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes	
:�*
dtype0
�
RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameRMSprop/dense_1/kernel/rms
�
.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms*
_output_shapes
:	�*
dtype0
�
RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_1/bias/rms
�
,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
�&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�%
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
�
,iter
	-decay
.learning_rate
/momentum
0rho	rms_	rms`	 rmsa	!rmsb	&rmsc	'rmsd
 
*
0
1
 2
!3
&4
'5
*
0
1
 2
!3
&4
'5
�

1layers
2layer_metrics
3layer_regularization_losses
	regularization_losses
4metrics
5non_trainable_variables

trainable_variables
	variables
 
 
 
 
�

6layers
7layer_metrics
8layer_regularization_losses
regularization_losses
9metrics
:non_trainable_variables
trainable_variables
	variables
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�

;layers
<layer_metrics
=layer_regularization_losses
regularization_losses
>metrics
?non_trainable_variables
trainable_variables
	variables
 
 
 
�

@layers
Alayer_metrics
Blayer_regularization_losses
regularization_losses
Cmetrics
Dnon_trainable_variables
trainable_variables
	variables
 
 
 
�

Elayers
Flayer_metrics
Glayer_regularization_losses
regularization_losses
Hmetrics
Inon_trainable_variables
trainable_variables
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
�

Jlayers
Klayer_metrics
Llayer_regularization_losses
"regularization_losses
Mmetrics
Nnon_trainable_variables
#trainable_variables
$	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
�

Olayers
Player_metrics
Qlayer_regularization_losses
(regularization_losses
Rmetrics
Snon_trainable_variables
)trainable_variables
*	variables
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6
 
 

T0
U1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Vtotal
	Wcount
X	variables
Y	keras_api
D
	Ztotal
	[count
\
_fn_kwargs
]	variables
^	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

X	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Z0
[1

]	variables
��
VARIABLE_VALUERMSprop/conv3d/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/conv3d/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/dense_1/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUERMSprop/dense_1/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*5
_output_shapes#
!:�����������*
dtype0**
shape!:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d/kernelconv3d/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_14281
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp-RMSprop/conv3d/kernel/rms/Read/ReadVariableOp+RMSprop/conv3d/bias/rms/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_14549
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/conv3d/kernel/rmsRMSprop/conv3d/bias/rmsRMSprop/dense/kernel/rmsRMSprop/dense/bias/rmsRMSprop/dense_1/kernel/rmsRMSprop/dense_1/bias/rms*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_14622��
�3
�
__inference__traced_save_14549
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop8
4savev2_rmsprop_conv3d_kernel_rms_read_readvariableop6
2savev2_rmsprop_conv3d_bias_rms_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_1_bias_rms_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_df0f39e2ba9e4f38bdb5a2df679164ff/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop4savev2_rmsprop_conv3d_kernel_rms_read_readvariableop2savev2_rmsprop_conv3d_bias_rms_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableop5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop3savev2_rmsprop_dense_1_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::Ϊ�:�:	�:: : : : : : : : : :::Ϊ�:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
::'#
!
_output_shapes
:Ϊ�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :0,
*
_output_shapes
:: 

_output_shapes
::'#
!
_output_shapes
:Ϊ�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�
�
&__inference_cnn_3d_layer_call_fn_14362

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_cnn_3d_layer_call_and_return_conditional_losses_142002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
�
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14153
input_1
conv3d_14079
conv3d_14081
dense_14121
dense_14123
dense_1_14147
dense_1_14149
identity��conv3d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_140492
rescaling/PartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv3d_14079conv3d_14081*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_140682 
conv3d/StatefulPartitionedCall�
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������OO* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_140302
max_pooling3d/PartitionedCall�
flatten/PartitionedCallPartitionedCall&max_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������Ϊ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_140912
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_14121dense_14123*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_141102
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14147dense_1_14149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_141362!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:�����������
!
_user_specified_name	input_1
�
{
&__inference_conv3d_layer_call_fn_14413

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_140682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:�����������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):�����������::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
�
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14239

inputs
conv3d_14221
conv3d_14223
dense_14228
dense_14230
dense_1_14233
dense_1_14235
identity��conv3d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_140492
rescaling/PartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv3d_14221conv3d_14223*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_140682 
conv3d/StatefulPartitionedCall�
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������OO* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_140302
max_pooling3d/PartitionedCall�
flatten/PartitionedCallPartitionedCall&max_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������Ϊ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_140912
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_14228dense_14230*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_141102
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14233dense_1_14235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_141362!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
�
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14345

inputs)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity�i
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <2
rescaling/Cast/xs
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2
rescaling/Cast_1/xy
rescaling/Cast_1Castrescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
rescaling/Cast_1�
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*5
_output_shapes#
!:�����������2
rescaling/mul�
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1:y:0*
T0*5
_output_shapes#
!:�����������2
rescaling/add�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02
conv3d/Conv3D/ReadVariableOp�
conv3d/Conv3DConv3Drescaling/add:z:0$conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������*
paddingVALID*
strides	
2
conv3d/Conv3D�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv3d/BiasAdd/ReadVariableOp�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������2
conv3d/BiasAdd{
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:�����������2
conv3d/Relu�
max_pooling3d/MaxPool3D	MaxPool3Dconv3d/Relu:activations:0*
T0*3
_output_shapes!
:���������OO*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d/MaxPool3Do
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����NU 2
flatten/Const�
flatten/ReshapeReshape max_pooling3d/MaxPool3D:output:0flatten/Const:output:0*
T0*)
_output_shapes
:���������Ϊ2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:Ϊ�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������:::::::] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_14281
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_140242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:�����������
!
_user_specified_name	input_1
�
�
@__inference_dense_layer_call_and_return_conditional_losses_14110

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:Ϊ�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:���������Ϊ:::Q M
)
_output_shapes
:���������Ϊ
 
_user_specified_nameinputs
�	
�
A__inference_conv3d_layer_call_and_return_conditional_losses_14068

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������*
paddingVALID*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�����������2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:�����������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):�����������:::] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�	
�
A__inference_conv3d_layer_call_and_return_conditional_losses_14404

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������*
paddingVALID*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�����������2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:�����������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):�����������:::] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_14136

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Y
�

!__inference__traced_restore_14622
file_prefix"
assignvariableop_conv3d_kernel"
assignvariableop_1_conv3d_bias#
assignvariableop_2_dense_kernel!
assignvariableop_3_dense_bias%
!assignvariableop_4_dense_1_kernel#
assignvariableop_5_dense_1_bias#
assignvariableop_6_rmsprop_iter$
 assignvariableop_7_rmsprop_decay,
(assignvariableop_8_rmsprop_learning_rate'
#assignvariableop_9_rmsprop_momentum#
assignvariableop_10_rmsprop_rho
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_11
-assignvariableop_15_rmsprop_conv3d_kernel_rms/
+assignvariableop_16_rmsprop_conv3d_bias_rms0
,assignvariableop_17_rmsprop_dense_kernel_rms.
*assignvariableop_18_rmsprop_dense_bias_rms2
.assignvariableop_19_rmsprop_dense_1_kernel_rms0
,assignvariableop_20_rmsprop_dense_1_bias_rms
identity_22��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_rmsprop_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp(assignvariableop_8_rmsprop_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_rmsprop_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_rmsprop_rhoIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_rmsprop_conv3d_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_rmsprop_conv3d_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp,assignvariableop_17_rmsprop_dense_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_rmsprop_dense_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_rmsprop_dense_1_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_rmsprop_dense_1_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21�
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
`
D__inference_rescaling_layer_call_and_return_conditional_losses_14049

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <2
Cast/x_
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1j
mulMulinputsCast/x:output:0*
T0*5
_output_shapes#
!:�����������2
mulh
addAddV2mul:z:0
Cast_1:y:0*
T0*5
_output_shapes#
!:�����������2
addi
IdentityIdentityadd:z:0*
T0*5
_output_shapes#
!:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_14091

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����NU 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:���������Ϊ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:���������Ϊ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������OO:[ W
3
_output_shapes!
:���������OO
 
_user_specified_nameinputs
�
�
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14200

inputs
conv3d_14182
conv3d_14184
dense_14189
dense_14191
dense_1_14194
dense_1_14196
identity��conv3d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_140492
rescaling/PartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv3d_14182conv3d_14184*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_140682 
conv3d/StatefulPartitionedCall�
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������OO* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_140302
max_pooling3d/PartitionedCall�
flatten/PartitionedCallPartitionedCall&max_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������Ϊ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_140912
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_14189dense_14191*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_141102
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14194dense_1_14196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_141362!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
E
)__inference_rescaling_layer_call_fn_14393

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_140492
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
�
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14313

inputs)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity�i
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <2
rescaling/Cast/xs
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2
rescaling/Cast_1/xy
rescaling/Cast_1Castrescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
rescaling/Cast_1�
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*5
_output_shapes#
!:�����������2
rescaling/mul�
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1:y:0*
T0*5
_output_shapes#
!:�����������2
rescaling/add�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02
conv3d/Conv3D/ReadVariableOp�
conv3d/Conv3DConv3Drescaling/add:z:0$conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������*
paddingVALID*
strides	
2
conv3d/Conv3D�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv3d/BiasAdd/ReadVariableOp�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������2
conv3d/BiasAdd{
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:�����������2
conv3d/Relu�
max_pooling3d/MaxPool3D	MaxPool3Dconv3d/Relu:activations:0*
T0*3
_output_shapes!
:���������OO*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d/MaxPool3Do
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����NU 2
flatten/Const�
flatten/ReshapeReshape max_pooling3d/MaxPool3D:output:0flatten/Const:output:0*
T0*)
_output_shapes
:���������Ϊ2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:Ϊ�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������:::::::] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
z
%__inference_dense_layer_call_fn_14444

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_141102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:���������Ϊ::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:���������Ϊ
 
_user_specified_nameinputs
�
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_14030

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
@__inference_dense_layer_call_and_return_conditional_losses_14435

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:Ϊ�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:���������Ϊ:::Q M
)
_output_shapes
:���������Ϊ
 
_user_specified_nameinputs
�
`
D__inference_rescaling_layer_call_and_return_conditional_losses_14388

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <2
Cast/x_
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1j
mulMulinputsCast/x:output:0*
T0*5
_output_shapes#
!:�����������2
mulh
addAddV2mul:z:0
Cast_1:y:0*
T0*5
_output_shapes#
!:�����������2
addi
IdentityIdentityadd:z:0*
T0*5
_output_shapes#
!:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
|
'__inference_dense_1_layer_call_fn_14463

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_141362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_14454

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_max_pooling3d_layer_call_fn_14036

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_140302
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
&__inference_cnn_3d_layer_call_fn_14379

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_cnn_3d_layer_call_and_return_conditional_losses_142392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs
�
�
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14175
input_1
conv3d_14157
conv3d_14159
dense_14164
dense_14166
dense_1_14169
dense_1_14171
identity��conv3d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_140492
rescaling/PartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv3d_14157conv3d_14159*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_140682 
conv3d/StatefulPartitionedCall�
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������OO* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_140302
max_pooling3d/PartitionedCall�
flatten/PartitionedCallPartitionedCall&max_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������Ϊ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_140912
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_14164dense_14166*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_141102
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14169dense_1_14171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_141362!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:�����������
!
_user_specified_name	input_1
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_14419

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����NU 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:���������Ϊ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:���������Ϊ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������OO:[ W
3
_output_shapes!
:���������OO
 
_user_specified_nameinputs
�
�
&__inference_cnn_3d_layer_call_fn_14215
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_cnn_3d_layer_call_and_return_conditional_losses_142002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:�����������
!
_user_specified_name	input_1
�
C
'__inference_flatten_layer_call_fn_14424

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������Ϊ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_140912
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:���������Ϊ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������OO:[ W
3
_output_shapes!
:���������OO
 
_user_specified_nameinputs
�#
�
 __inference__wrapped_model_14024
input_10
,cnn_3d_conv3d_conv3d_readvariableop_resource1
-cnn_3d_conv3d_biasadd_readvariableop_resource/
+cnn_3d_dense_matmul_readvariableop_resource0
,cnn_3d_dense_biasadd_readvariableop_resource1
-cnn_3d_dense_1_matmul_readvariableop_resource2
.cnn_3d_dense_1_biasadd_readvariableop_resource
identity�w
cnn_3d/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <2
cnn_3d/rescaling/Cast/x�
cnn_3d/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2
cnn_3d/rescaling/Cast_1/x�
cnn_3d/rescaling/Cast_1Cast"cnn_3d/rescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
cnn_3d/rescaling/Cast_1�
cnn_3d/rescaling/mulMulinput_1 cnn_3d/rescaling/Cast/x:output:0*
T0*5
_output_shapes#
!:�����������2
cnn_3d/rescaling/mul�
cnn_3d/rescaling/addAddV2cnn_3d/rescaling/mul:z:0cnn_3d/rescaling/Cast_1:y:0*
T0*5
_output_shapes#
!:�����������2
cnn_3d/rescaling/add�
#cnn_3d/conv3d/Conv3D/ReadVariableOpReadVariableOp,cnn_3d_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02%
#cnn_3d/conv3d/Conv3D/ReadVariableOp�
cnn_3d/conv3d/Conv3DConv3Dcnn_3d/rescaling/add:z:0+cnn_3d/conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������*
paddingVALID*
strides	
2
cnn_3d/conv3d/Conv3D�
$cnn_3d/conv3d/BiasAdd/ReadVariableOpReadVariableOp-cnn_3d_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$cnn_3d/conv3d/BiasAdd/ReadVariableOp�
cnn_3d/conv3d/BiasAddBiasAddcnn_3d/conv3d/Conv3D:output:0,cnn_3d/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�����������2
cnn_3d/conv3d/BiasAdd�
cnn_3d/conv3d/ReluRelucnn_3d/conv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:�����������2
cnn_3d/conv3d/Relu�
cnn_3d/max_pooling3d/MaxPool3D	MaxPool3D cnn_3d/conv3d/Relu:activations:0*
T0*3
_output_shapes!
:���������OO*
ksize	
*
paddingVALID*
strides	
2 
cnn_3d/max_pooling3d/MaxPool3D}
cnn_3d/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����NU 2
cnn_3d/flatten/Const�
cnn_3d/flatten/ReshapeReshape'cnn_3d/max_pooling3d/MaxPool3D:output:0cnn_3d/flatten/Const:output:0*
T0*)
_output_shapes
:���������Ϊ2
cnn_3d/flatten/Reshape�
"cnn_3d/dense/MatMul/ReadVariableOpReadVariableOp+cnn_3d_dense_matmul_readvariableop_resource*!
_output_shapes
:Ϊ�*
dtype02$
"cnn_3d/dense/MatMul/ReadVariableOp�
cnn_3d/dense/MatMulMatMulcnn_3d/flatten/Reshape:output:0*cnn_3d/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
cnn_3d/dense/MatMul�
#cnn_3d/dense/BiasAdd/ReadVariableOpReadVariableOp,cnn_3d_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#cnn_3d/dense/BiasAdd/ReadVariableOp�
cnn_3d/dense/BiasAddBiasAddcnn_3d/dense/MatMul:product:0+cnn_3d/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
cnn_3d/dense/BiasAdd�
cnn_3d/dense/ReluRelucnn_3d/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
cnn_3d/dense/Relu�
$cnn_3d/dense_1/MatMul/ReadVariableOpReadVariableOp-cnn_3d_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02&
$cnn_3d/dense_1/MatMul/ReadVariableOp�
cnn_3d/dense_1/MatMulMatMulcnn_3d/dense/Relu:activations:0,cnn_3d/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
cnn_3d/dense_1/MatMul�
%cnn_3d/dense_1/BiasAdd/ReadVariableOpReadVariableOp.cnn_3d_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%cnn_3d/dense_1/BiasAdd/ReadVariableOp�
cnn_3d/dense_1/BiasAddBiasAddcnn_3d/dense_1/MatMul:product:0-cnn_3d/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
cnn_3d/dense_1/BiasAdds
IdentityIdentitycnn_3d/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������:::::::^ Z
5
_output_shapes#
!:�����������
!
_user_specified_name	input_1
�
�
&__inference_cnn_3d_layer_call_fn_14254
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_cnn_3d_layer_call_and_return_conditional_losses_142392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:�����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:�����������
!
_user_specified_name	input_1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
input_1>
serving_default_input_1:0�����������;
dense_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�5
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
*e&call_and_return_all_conditional_losses
f__call__
g_default_save_signature"�3
_tf_keras_network�2{"class_name": "Functional", "name": "cnn_3d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_3d", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 160, 160, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Rescaling", "config": {"name": "rescaling", "trainable": true, "dtype": "float32", "scale": 0.00784313725490196, "offset": -1}, "name": "rescaling", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["rescaling", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "max_pooling3d", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling3d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 160, 160, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "cnn_3d", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 160, 160, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Rescaling", "config": {"name": "rescaling", "trainable": true, "dtype": "float32", "scale": 0.00784313725490196, "offset": -1}, "name": "rescaling", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["rescaling", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "max_pooling3d", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling3d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 160, 160, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 160, 160, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�
regularization_losses
trainable_variables
	variables
	keras_api
*h&call_and_return_all_conditional_losses
i__call__"�
_tf_keras_layer�{"class_name": "Rescaling", "name": "rescaling", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rescaling", "trainable": true, "dtype": "float32", "scale": 0.00784313725490196, "offset": -1}}
�	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*j&call_and_return_all_conditional_losses
k__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 160, 160, 3]}}
�
regularization_losses
trainable_variables
	variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"�
_tf_keras_layer�{"class_name": "MaxPooling3D", "name": "max_pooling3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
regularization_losses
trainable_variables
	variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
*p&call_and_return_all_conditional_losses
q__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 87374}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 87374]}}
�

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
*r&call_and_return_all_conditional_losses
s__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�
,iter
	-decay
.learning_rate
/momentum
0rho	rms_	rms`	 rmsa	!rmsb	&rmsc	'rmsd"
	optimizer
 "
trackable_list_wrapper
J
0
1
 2
!3
&4
'5"
trackable_list_wrapper
J
0
1
 2
!3
&4
'5"
trackable_list_wrapper
�

1layers
2layer_metrics
3layer_regularization_losses
	regularization_losses
4metrics
5non_trainable_variables

trainable_variables
	variables
f__call__
g_default_save_signature
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
,
tserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

6layers
7layer_metrics
8layer_regularization_losses
regularization_losses
9metrics
:non_trainable_variables
trainable_variables
	variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
+:)2conv3d/kernel
:2conv3d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

;layers
<layer_metrics
=layer_regularization_losses
regularization_losses
>metrics
?non_trainable_variables
trainable_variables
	variables
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

@layers
Alayer_metrics
Blayer_regularization_losses
regularization_losses
Cmetrics
Dnon_trainable_variables
trainable_variables
	variables
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

Elayers
Flayer_metrics
Glayer_regularization_losses
regularization_losses
Hmetrics
Inon_trainable_variables
trainable_variables
	variables
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
!:Ϊ�2dense/kernel
:�2
dense/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
�

Jlayers
Klayer_metrics
Llayer_regularization_losses
"regularization_losses
Mmetrics
Nnon_trainable_variables
#trainable_variables
$	variables
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
!:	�2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�

Olayers
Player_metrics
Qlayer_regularization_losses
(regularization_losses
Rmetrics
Snon_trainable_variables
)trainable_variables
*	variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	Vtotal
	Wcount
X	variables
Y	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	Ztotal
	[count
\
_fn_kwargs
]	variables
^	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
V0
W1"
trackable_list_wrapper
-
X	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
5:32RMSprop/conv3d/kernel/rms
#:!2RMSprop/conv3d/bias/rms
+:)Ϊ�2RMSprop/dense/kernel/rms
#:!�2RMSprop/dense/bias/rms
+:)	�2RMSprop/dense_1/kernel/rms
$:"2RMSprop/dense_1/bias/rms
�2�
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14345
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14313
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14175
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14153�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_cnn_3d_layer_call_fn_14254
&__inference_cnn_3d_layer_call_fn_14362
&__inference_cnn_3d_layer_call_fn_14215
&__inference_cnn_3d_layer_call_fn_14379�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_14024�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *4�1
/�,
input_1�����������
�2�
D__inference_rescaling_layer_call_and_return_conditional_losses_14388�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_rescaling_layer_call_fn_14393�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_conv3d_layer_call_and_return_conditional_losses_14404�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_conv3d_layer_call_fn_14413�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_14030�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *M�J
H�EA���������������������������������������������
�2�
-__inference_max_pooling3d_layer_call_fn_14036�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *M�J
H�EA���������������������������������������������
�2�
B__inference_flatten_layer_call_and_return_conditional_losses_14419�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_flatten_layer_call_fn_14424�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dense_layer_call_and_return_conditional_losses_14435�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_layer_call_fn_14444�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_1_layer_call_and_return_conditional_losses_14454�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_1_layer_call_fn_14463�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2B0
#__inference_signature_wrapper_14281input_1�
 __inference__wrapped_model_14024{ !&'>�;
4�1
/�,
input_1�����������
� "1�.
,
dense_1!�
dense_1����������
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14153w !&'F�C
<�9
/�,
input_1�����������
p

 
� "%�"
�
0���������
� �
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14175w !&'F�C
<�9
/�,
input_1�����������
p 

 
� "%�"
�
0���������
� �
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14313v !&'E�B
;�8
.�+
inputs�����������
p

 
� "%�"
�
0���������
� �
A__inference_cnn_3d_layer_call_and_return_conditional_losses_14345v !&'E�B
;�8
.�+
inputs�����������
p 

 
� "%�"
�
0���������
� �
&__inference_cnn_3d_layer_call_fn_14215j !&'F�C
<�9
/�,
input_1�����������
p

 
� "�����������
&__inference_cnn_3d_layer_call_fn_14254j !&'F�C
<�9
/�,
input_1�����������
p 

 
� "�����������
&__inference_cnn_3d_layer_call_fn_14362i !&'E�B
;�8
.�+
inputs�����������
p

 
� "�����������
&__inference_cnn_3d_layer_call_fn_14379i !&'E�B
;�8
.�+
inputs�����������
p 

 
� "�����������
A__inference_conv3d_layer_call_and_return_conditional_losses_14404x=�:
3�0
.�+
inputs�����������
� "3�0
)�&
0�����������
� �
&__inference_conv3d_layer_call_fn_14413k=�:
3�0
.�+
inputs�����������
� "&�#������������
B__inference_dense_1_layer_call_and_return_conditional_losses_14454]&'0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_dense_1_layer_call_fn_14463P&'0�-
&�#
!�
inputs����������
� "�����������
@__inference_dense_layer_call_and_return_conditional_losses_14435_ !1�.
'�$
"�
inputs���������Ϊ
� "&�#
�
0����������
� {
%__inference_dense_layer_call_fn_14444R !1�.
'�$
"�
inputs���������Ϊ
� "������������
B__inference_flatten_layer_call_and_return_conditional_losses_14419f;�8
1�.
,�)
inputs���������OO
� "'�$
�
0���������Ϊ
� �
'__inference_flatten_layer_call_fn_14424Y;�8
1�.
,�)
inputs���������OO
� "����������Ϊ�
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_14030�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
-__inference_max_pooling3d_layer_call_fn_14036�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
D__inference_rescaling_layer_call_and_return_conditional_losses_14388t=�:
3�0
.�+
inputs�����������
� "3�0
)�&
0�����������
� �
)__inference_rescaling_layer_call_fn_14393g=�:
3�0
.�+
inputs�����������
� "&�#������������
#__inference_signature_wrapper_14281� !&'I�F
� 
?�<
:
input_1/�,
input_1�����������"1�.
,
dense_1!�
dense_1���������