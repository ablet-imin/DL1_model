??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8̢
}
maxout0/maxout_wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	)?*!
shared_namemaxout0/maxout_w
v
$maxout0/maxout_w/Read/ReadVariableOpReadVariableOpmaxout0/maxout_w*
_output_shapes
:	)?*
dtype0
y
maxout0/maxout_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namemaxout0/maxout_b
r
$maxout0/maxout_b/Read/ReadVariableOpReadVariableOpmaxout0/maxout_b*
_output_shapes	
:?*
dtype0
p

BN1/norm_wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:HH*
shared_name
BN1/norm_w
i
BN1/norm_w/Read/ReadVariableOpReadVariableOp
BN1/norm_w*
_output_shapes

:HH*
dtype0
l

BN1/norm_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_name
BN1/norm_b
e
BN1/norm_b/Read/ReadVariableOpReadVariableOp
BN1/norm_b*
_output_shapes
:H*
dtype0
v
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H9*
shared_namedense2/kernel
o
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes

:H9*
dtype0
n
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:9*
dtype0
p

BN3/norm_wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:99*
shared_name
BN3/norm_w
i
BN3/norm_w/Read/ReadVariableOpReadVariableOp
BN3/norm_w*
_output_shapes

:99*
dtype0
l

BN3/norm_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*
shared_name
BN3/norm_b
e
BN3/norm_b/Read/ReadVariableOpReadVariableOp
BN3/norm_b*
_output_shapes
:9*
dtype0
v
dense4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:9<*
shared_namedense4/kernel
o
!dense4/kernel/Read/ReadVariableOpReadVariableOpdense4/kernel*
_output_shapes

:9<*
dtype0
n
dense4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namedense4/bias
g
dense4/bias/Read/ReadVariableOpReadVariableOpdense4/bias*
_output_shapes
:<*
dtype0
p

BN5/norm_wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*
shared_name
BN5/norm_w
i
BN5/norm_w/Read/ReadVariableOpReadVariableOp
BN5/norm_w*
_output_shapes

:<<*
dtype0
l

BN5/norm_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_name
BN5/norm_b
e
BN5/norm_b/Read/ReadVariableOpReadVariableOp
BN5/norm_b*
_output_shapes
:<*
dtype0
v
dense6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<0*
shared_namedense6/kernel
o
!dense6/kernel/Read/ReadVariableOpReadVariableOpdense6/kernel*
_output_shapes

:<0*
dtype0
n
dense6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense6/bias
g
dense6/bias/Read/ReadVariableOpReadVariableOpdense6/bias*
_output_shapes
:0*
dtype0
p

BN7/norm_wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:00*
shared_name
BN7/norm_w
i
BN7/norm_w/Read/ReadVariableOpReadVariableOp
BN7/norm_w*
_output_shapes

:00*
dtype0
l

BN7/norm_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_name
BN7/norm_b
e
BN7/norm_b/Read/ReadVariableOpReadVariableOp
BN7/norm_b*
_output_shapes
:0*
dtype0
v
dense8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0$*
shared_namedense8/kernel
o
!dense8/kernel/Read/ReadVariableOpReadVariableOpdense8/kernel*
_output_shapes

:0$*
dtype0
n
dense8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_namedense8/bias
g
dense8/bias/Read/ReadVariableOpReadVariableOpdense8/bias*
_output_shapes
:$*
dtype0
p

BN9/norm_wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*
shared_name
BN9/norm_w
i
BN9/norm_w/Read/ReadVariableOpReadVariableOp
BN9/norm_w*
_output_shapes

:$$*
dtype0
l

BN9/norm_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_name
BN9/norm_b
e
BN9/norm_b/Read/ReadVariableOpReadVariableOp
BN9/norm_b*
_output_shapes
:$*
dtype0

maxout10/maxout_wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	$?*"
shared_namemaxout10/maxout_w
x
%maxout10/maxout_w/Read/ReadVariableOpReadVariableOpmaxout10/maxout_w*
_output_shapes
:	$?*
dtype0
{
maxout10/maxout_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namemaxout10/maxout_b
t
%maxout10/maxout_b/Read/ReadVariableOpReadVariableOpmaxout10/maxout_b*
_output_shapes	
:?*
dtype0
r
BN11/norm_wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameBN11/norm_w
k
BN11/norm_w/Read/ReadVariableOpReadVariableOpBN11/norm_w*
_output_shapes

:*
dtype0
n
BN11/norm_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameBN11/norm_b
g
BN11/norm_b/Read/ReadVariableOpReadVariableOpBN11/norm_b*
_output_shapes
:*
dtype0
x
dense12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense12/kernel
q
"dense12/kernel/Read/ReadVariableOpReadVariableOpdense12/kernel*
_output_shapes

:*
dtype0
p
dense12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense12/bias
i
 dense12/bias/Read/ReadVariableOpReadVariableOpdense12/bias*
_output_shapes
:*
dtype0
r
BN13/norm_wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameBN13/norm_w
k
BN13/norm_w/Read/ReadVariableOpReadVariableOpBN13/norm_w*
_output_shapes

:*
dtype0
n
BN13/norm_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameBN13/norm_b
g
BN13/norm_b/Read/ReadVariableOpReadVariableOpBN13/norm_b*
_output_shapes
:*
dtype0
x
dense14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense14/kernel
q
"dense14/kernel/Read/ReadVariableOpReadVariableOpdense14/kernel*
_output_shapes

:*
dtype0
p
dense14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense14/bias
i
 dense14/bias/Read/ReadVariableOpReadVariableOpdense14/bias*
_output_shapes
:*
dtype0
r
BN15/norm_wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameBN15/norm_w
k
BN15/norm_w/Read/ReadVariableOpReadVariableOpBN15/norm_w*
_output_shapes

:*
dtype0
n
BN15/norm_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameBN15/norm_b
g
BN15/norm_b/Read/ReadVariableOpReadVariableOpBN15/norm_b*
_output_shapes
:*
dtype0
x
dense16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense16/kernel
q
"dense16/kernel/Read/ReadVariableOpReadVariableOpdense16/kernel*
_output_shapes

:*
dtype0
p
dense16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense16/bias
i
 dense16/bias/Read/ReadVariableOpReadVariableOpdense16/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?R
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?R
value?RB?R B?R
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
layer_with_weights-16
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
|
maxout_w
w
maxout_b
b
	variables
trainable_variables
regularization_losses
	keras_api
x

norm_w
w

 norm_b
 b
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
x

+norm_w
+w

,norm_b
,b
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
x

7norm_w
7w

8norm_b
8b
9	variables
:trainable_variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
x

Cnorm_w
Cw

Dnorm_b
Db
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
x

Onorm_w
Ow

Pnorm_b
Pb
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
|
Umaxout_w
Uw
Vmaxout_b
Vb
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
x

[norm_w
[w

\norm_b
\b
]	variables
^trainable_variables
_regularization_losses
`	keras_api
h

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
x

gnorm_w
gw

hnorm_b
hb
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
h

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
x

snorm_w
sw

tnorm_b
tb
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
h

ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
 
?
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
I16
J17
O18
P19
U20
V21
[22
\23
a24
b25
g26
h27
m28
n29
s30
t31
y32
z33
?
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
I16
J17
O18
P19
U20
V21
[22
\23
a24
b25
g26
h27
m28
n29
s30
t31
y32
z33
 
?
layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
	variables
?metrics
trainable_variables
regularization_losses
 
^\
VARIABLE_VALUEmaxout0/maxout_w8layer_with_weights-0/maxout_w/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEmaxout0/maxout_b8layer_with_weights-0/maxout_b/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
	variables
?metrics
trainable_variables
regularization_losses
VT
VARIABLE_VALUE
BN1/norm_w6layer_with_weights-1/norm_w/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUE
BN1/norm_b6layer_with_weights-1/norm_b/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
!	variables
?metrics
"trainable_variables
#regularization_losses
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
'	variables
?metrics
(trainable_variables
)regularization_losses
VT
VARIABLE_VALUE
BN3/norm_w6layer_with_weights-3/norm_w/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUE
BN3/norm_b6layer_with_weights-3/norm_b/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
-	variables
?metrics
.trainable_variables
/regularization_losses
YW
VARIABLE_VALUEdense4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
3	variables
?metrics
4trainable_variables
5regularization_losses
VT
VARIABLE_VALUE
BN5/norm_w6layer_with_weights-5/norm_w/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUE
BN5/norm_b6layer_with_weights-5/norm_b/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
9	variables
?metrics
:trainable_variables
;regularization_losses
YW
VARIABLE_VALUEdense6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
?	variables
?metrics
@trainable_variables
Aregularization_losses
VT
VARIABLE_VALUE
BN7/norm_w6layer_with_weights-7/norm_w/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUE
BN7/norm_b6layer_with_weights-7/norm_b/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
E	variables
?metrics
Ftrainable_variables
Gregularization_losses
YW
VARIABLE_VALUEdense8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
K	variables
?metrics
Ltrainable_variables
Mregularization_losses
VT
VARIABLE_VALUE
BN9/norm_w6layer_with_weights-9/norm_w/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUE
BN9/norm_b6layer_with_weights-9/norm_b/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

O0
P1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
Q	variables
?metrics
Rtrainable_variables
Sregularization_losses
`^
VARIABLE_VALUEmaxout10/maxout_w9layer_with_weights-10/maxout_w/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEmaxout10/maxout_b9layer_with_weights-10/maxout_b/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
W	variables
?metrics
Xtrainable_variables
Yregularization_losses
XV
VARIABLE_VALUEBN11/norm_w7layer_with_weights-11/norm_w/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEBN11/norm_b7layer_with_weights-11/norm_b/.ATTRIBUTES/VARIABLE_VALUE

[0
\1

[0
\1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
]	variables
?metrics
^trainable_variables
_regularization_losses
[Y
VARIABLE_VALUEdense12/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense12/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

a0
b1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
c	variables
?metrics
dtrainable_variables
eregularization_losses
XV
VARIABLE_VALUEBN13/norm_w7layer_with_weights-13/norm_w/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEBN13/norm_b7layer_with_weights-13/norm_b/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
i	variables
?metrics
jtrainable_variables
kregularization_losses
[Y
VARIABLE_VALUEdense14/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense14/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

m0
n1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
o	variables
?metrics
ptrainable_variables
qregularization_losses
XV
VARIABLE_VALUEBN15/norm_w7layer_with_weights-15/norm_w/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEBN15/norm_b7layer_with_weights-15/norm_b/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
u	variables
?metrics
vtrainable_variables
wregularization_losses
[Y
VARIABLE_VALUEdense16/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense16/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

y0
z1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
{	variables
?metrics
|trainable_variables
}regularization_losses
 
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
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
x
serving_default_inputPlaceholder*'
_output_shapes
:?????????)*
dtype0*
shape:?????????)
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputmaxout0/maxout_wmaxout0/maxout_b
BN1/norm_b
BN1/norm_wdense2/kerneldense2/bias
BN3/norm_b
BN3/norm_wdense4/kerneldense4/bias
BN5/norm_b
BN5/norm_wdense6/kerneldense6/bias
BN7/norm_b
BN7/norm_wdense8/kerneldense8/bias
BN9/norm_b
BN9/norm_wmaxout10/maxout_wmaxout10/maxout_bBN11/norm_bBN11/norm_wdense12/kerneldense12/biasBN13/norm_bBN13/norm_wdense14/kerneldense14/biasBN15/norm_bBN15/norm_wdense16/kerneldense16/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_1656
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$maxout0/maxout_w/Read/ReadVariableOp$maxout0/maxout_b/Read/ReadVariableOpBN1/norm_w/Read/ReadVariableOpBN1/norm_b/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOpBN3/norm_w/Read/ReadVariableOpBN3/norm_b/Read/ReadVariableOp!dense4/kernel/Read/ReadVariableOpdense4/bias/Read/ReadVariableOpBN5/norm_w/Read/ReadVariableOpBN5/norm_b/Read/ReadVariableOp!dense6/kernel/Read/ReadVariableOpdense6/bias/Read/ReadVariableOpBN7/norm_w/Read/ReadVariableOpBN7/norm_b/Read/ReadVariableOp!dense8/kernel/Read/ReadVariableOpdense8/bias/Read/ReadVariableOpBN9/norm_w/Read/ReadVariableOpBN9/norm_b/Read/ReadVariableOp%maxout10/maxout_w/Read/ReadVariableOp%maxout10/maxout_b/Read/ReadVariableOpBN11/norm_w/Read/ReadVariableOpBN11/norm_b/Read/ReadVariableOp"dense12/kernel/Read/ReadVariableOp dense12/bias/Read/ReadVariableOpBN13/norm_w/Read/ReadVariableOpBN13/norm_b/Read/ReadVariableOp"dense14/kernel/Read/ReadVariableOp dense14/bias/Read/ReadVariableOpBN15/norm_w/Read/ReadVariableOpBN15/norm_b/Read/ReadVariableOp"dense16/kernel/Read/ReadVariableOp dense16/bias/Read/ReadVariableOpConst*/
Tin(
&2$*
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
GPU 2J 8? *&
f!R
__inference__traced_save_2579
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemaxout0/maxout_wmaxout0/maxout_b
BN1/norm_w
BN1/norm_bdense2/kerneldense2/bias
BN3/norm_w
BN3/norm_bdense4/kerneldense4/bias
BN5/norm_w
BN5/norm_bdense6/kerneldense6/bias
BN7/norm_w
BN7/norm_bdense8/kerneldense8/bias
BN9/norm_w
BN9/norm_bmaxout10/maxout_wmaxout10/maxout_bBN11/norm_wBN11/norm_bdense12/kerneldense12/biasBN13/norm_wBN13/norm_bdense14/kerneldense14/biasBN15/norm_wBN15/norm_bdense16/kerneldense16/bias*.
Tin'
%2#*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_2691??
?P
?	
?__inference_model_layer_call_and_return_conditional_losses_1256	
input
maxout0_1170
maxout0_1172
bn1_1175
bn1_1177
dense2_1180
dense2_1182
bn3_1185
bn3_1187
dense4_1190
dense4_1192
bn5_1195
bn5_1197
dense6_1200
dense6_1202
bn7_1205
bn7_1207
dense8_1210
dense8_1212
bn9_1215
bn9_1217
maxout10_1220
maxout10_1222
	bn11_1225
	bn11_1227
dense12_1230
dense12_1232
	bn13_1235
	bn13_1237
dense14_1240
dense14_1242
	bn15_1245
	bn15_1247
dense16_1250
dense16_1252
identity??BN1/StatefulPartitionedCall?BN11/StatefulPartitionedCall?BN13/StatefulPartitionedCall?BN15/StatefulPartitionedCall?BN3/StatefulPartitionedCall?BN5/StatefulPartitionedCall?BN7/StatefulPartitionedCall?BN9/StatefulPartitionedCall?dense12/StatefulPartitionedCall?dense14/StatefulPartitionedCall?dense16/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense4/StatefulPartitionedCall?dense6/StatefulPartitionedCall?dense8/StatefulPartitionedCall?maxout0/StatefulPartitionedCall? maxout10/StatefulPartitionedCall?
maxout0/StatefulPartitionedCallStatefulPartitionedCallinputmaxout0_1170maxout0_1172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_maxout0_layer_call_and_return_conditional_losses_7112!
maxout0/StatefulPartitionedCall?
BN1/StatefulPartitionedCallStatefulPartitionedCall(maxout0/StatefulPartitionedCall:output:0bn1_1175bn1_1177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN1_layer_call_and_return_conditional_losses_7372
BN1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall$BN1/StatefulPartitionedCall:output:0dense2_1180dense2_1182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense2_layer_call_and_return_conditional_losses_7642 
dense2/StatefulPartitionedCall?
BN3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0bn3_1185bn3_1187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN3_layer_call_and_return_conditional_losses_7902
BN3/StatefulPartitionedCall?
dense4/StatefulPartitionedCallStatefulPartitionedCall$BN3/StatefulPartitionedCall:output:0dense4_1190dense4_1192*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense4_layer_call_and_return_conditional_losses_8172 
dense4/StatefulPartitionedCall?
BN5/StatefulPartitionedCallStatefulPartitionedCall'dense4/StatefulPartitionedCall:output:0bn5_1195bn5_1197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN5_layer_call_and_return_conditional_losses_8432
BN5/StatefulPartitionedCall?
dense6/StatefulPartitionedCallStatefulPartitionedCall$BN5/StatefulPartitionedCall:output:0dense6_1200dense6_1202*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense6_layer_call_and_return_conditional_losses_8702 
dense6/StatefulPartitionedCall?
BN7/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0bn7_1205bn7_1207*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN7_layer_call_and_return_conditional_losses_8962
BN7/StatefulPartitionedCall?
dense8/StatefulPartitionedCallStatefulPartitionedCall$BN7/StatefulPartitionedCall:output:0dense8_1210dense8_1212*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense8_layer_call_and_return_conditional_losses_9232 
dense8/StatefulPartitionedCall?
BN9/StatefulPartitionedCallStatefulPartitionedCall'dense8/StatefulPartitionedCall:output:0bn9_1215bn9_1217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN9_layer_call_and_return_conditional_losses_9492
BN9/StatefulPartitionedCall?
 maxout10/StatefulPartitionedCallStatefulPartitionedCall$BN9/StatefulPartitionedCall:output:0maxout10_1220maxout10_1222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_maxout10_layer_call_and_return_conditional_losses_9912"
 maxout10/StatefulPartitionedCall?
BN11/StatefulPartitionedCallStatefulPartitionedCall)maxout10/StatefulPartitionedCall:output:0	bn11_1225	bn11_1227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN11_layer_call_and_return_conditional_losses_10172
BN11/StatefulPartitionedCall?
dense12/StatefulPartitionedCallStatefulPartitionedCall%BN11/StatefulPartitionedCall:output:0dense12_1230dense12_1232*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense12_layer_call_and_return_conditional_losses_10442!
dense12/StatefulPartitionedCall?
BN13/StatefulPartitionedCallStatefulPartitionedCall(dense12/StatefulPartitionedCall:output:0	bn13_1235	bn13_1237*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN13_layer_call_and_return_conditional_losses_10702
BN13/StatefulPartitionedCall?
dense14/StatefulPartitionedCallStatefulPartitionedCall%BN13/StatefulPartitionedCall:output:0dense14_1240dense14_1242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense14_layer_call_and_return_conditional_losses_10972!
dense14/StatefulPartitionedCall?
BN15/StatefulPartitionedCallStatefulPartitionedCall(dense14/StatefulPartitionedCall:output:0	bn15_1245	bn15_1247*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN15_layer_call_and_return_conditional_losses_11232
BN15/StatefulPartitionedCall?
dense16/StatefulPartitionedCallStatefulPartitionedCall%BN15/StatefulPartitionedCall:output:0dense16_1250dense16_1252*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense16_layer_call_and_return_conditional_losses_11502!
dense16/StatefulPartitionedCall?
IdentityIdentity(dense16/StatefulPartitionedCall:output:0^BN1/StatefulPartitionedCall^BN11/StatefulPartitionedCall^BN13/StatefulPartitionedCall^BN15/StatefulPartitionedCall^BN3/StatefulPartitionedCall^BN5/StatefulPartitionedCall^BN7/StatefulPartitionedCall^BN9/StatefulPartitionedCall ^dense12/StatefulPartitionedCall ^dense14/StatefulPartitionedCall ^dense16/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense4/StatefulPartitionedCall^dense6/StatefulPartitionedCall^dense8/StatefulPartitionedCall ^maxout0/StatefulPartitionedCall!^maxout10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::2:
BN1/StatefulPartitionedCallBN1/StatefulPartitionedCall2<
BN11/StatefulPartitionedCallBN11/StatefulPartitionedCall2<
BN13/StatefulPartitionedCallBN13/StatefulPartitionedCall2<
BN15/StatefulPartitionedCallBN15/StatefulPartitionedCall2:
BN3/StatefulPartitionedCallBN3/StatefulPartitionedCall2:
BN5/StatefulPartitionedCallBN5/StatefulPartitionedCall2:
BN7/StatefulPartitionedCallBN7/StatefulPartitionedCall2:
BN9/StatefulPartitionedCallBN9/StatefulPartitionedCall2B
dense12/StatefulPartitionedCalldense12/StatefulPartitionedCall2B
dense14/StatefulPartitionedCalldense14/StatefulPartitionedCall2B
dense16/StatefulPartitionedCalldense16/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall2@
dense8/StatefulPartitionedCalldense8/StatefulPartitionedCall2B
maxout0/StatefulPartitionedCallmaxout0/StatefulPartitionedCall2D
 maxout10/StatefulPartitionedCall maxout10/StatefulPartitionedCall:N J
'
_output_shapes
:?????????)

_user_specified_nameinput
?
w
"__inference_BN9_layer_call_fn_2302

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN9_layer_call_and_return_conditional_losses_9492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_1581	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_15102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????)

_user_specified_nameinput
?
?
=__inference_BN7_layer_call_and_return_conditional_losses_2254

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:0*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:00*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
w
"__inference_BN7_layer_call_fn_2263

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN7_layer_call_and_return_conditional_losses_8962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
<__inference_BN9_layer_call_and_return_conditional_losses_949

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:$*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs
?
{
&__inference_dense12_layer_call_fn_2376

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense12_layer_call_and_return_conditional_losses_10442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
w
"__inference_BN3_layer_call_fn_2185

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN3_layer_call_and_return_conditional_losses_7902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????92

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
>__inference_BN13_layer_call_and_return_conditional_losses_2386

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_dense16_layer_call_and_return_conditional_losses_2445

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_dense16_layer_call_and_return_conditional_losses_1150

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_maxout0_layer_call_and_return_conditional_losses_711

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	)?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpt
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
adds
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   H   2
Reshape/shapet
ReshapeReshapeadd:z:0Reshape/shape:output:0*
T0*+
_output_shapes
:?????????H2	
Reshape\
maxout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:2
maxout/Shape?
maxout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
maxout/strided_slice/stack?
maxout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
maxout/strided_slice/stack_1?
maxout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
maxout/strided_slice/stack_2?
maxout/strided_sliceStridedSlicemaxout/Shape:output:0#maxout/strided_slice/stack:output:0%maxout/strided_slice/stack_1:output:0%maxout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
maxout/strided_slicer
maxout/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/1r
maxout/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/2r
maxout/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :H2
maxout/Reshape/shape/3?
maxout/Reshape/shapePackmaxout/strided_slice:output:0maxout/Reshape/shape/1:output:0maxout/Reshape/shape/2:output:0maxout/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
maxout/Reshape/shape?
maxout/ReshapeReshapeReshape:output:0maxout/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????H2
maxout/Reshape~
maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
maxout/Max/reduction_indices?

maxout/MaxMaxmaxout/Reshape:output:0%maxout/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????H2

maxout/Maxs
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????H   2
Reshape_1/shape?
	Reshape_1Reshapemaxout/Max:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????H2
	Reshape_1?
IdentityIdentityReshape_1:output:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????)::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?E
?
__inference__traced_save_2579
file_prefix/
+savev2_maxout0_maxout_w_read_readvariableop/
+savev2_maxout0_maxout_b_read_readvariableop)
%savev2_bn1_norm_w_read_readvariableop)
%savev2_bn1_norm_b_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop)
%savev2_bn3_norm_w_read_readvariableop)
%savev2_bn3_norm_b_read_readvariableop,
(savev2_dense4_kernel_read_readvariableop*
&savev2_dense4_bias_read_readvariableop)
%savev2_bn5_norm_w_read_readvariableop)
%savev2_bn5_norm_b_read_readvariableop,
(savev2_dense6_kernel_read_readvariableop*
&savev2_dense6_bias_read_readvariableop)
%savev2_bn7_norm_w_read_readvariableop)
%savev2_bn7_norm_b_read_readvariableop,
(savev2_dense8_kernel_read_readvariableop*
&savev2_dense8_bias_read_readvariableop)
%savev2_bn9_norm_w_read_readvariableop)
%savev2_bn9_norm_b_read_readvariableop0
,savev2_maxout10_maxout_w_read_readvariableop0
,savev2_maxout10_maxout_b_read_readvariableop*
&savev2_bn11_norm_w_read_readvariableop*
&savev2_bn11_norm_b_read_readvariableop-
)savev2_dense12_kernel_read_readvariableop+
'savev2_dense12_bias_read_readvariableop*
&savev2_bn13_norm_w_read_readvariableop*
&savev2_bn13_norm_b_read_readvariableop-
)savev2_dense14_kernel_read_readvariableop+
'savev2_dense14_bias_read_readvariableop*
&savev2_bn15_norm_w_read_readvariableop*
&savev2_bn15_norm_b_read_readvariableop-
)savev2_dense16_kernel_read_readvariableop+
'savev2_dense16_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B8layer_with_weights-0/maxout_w/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/maxout_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/norm_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/norm_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/norm_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/norm_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/norm_b/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-10/maxout_w/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-10/maxout_b/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/norm_w/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/norm_b/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/norm_w/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/norm_b/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/norm_w/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/norm_b/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_maxout0_maxout_w_read_readvariableop+savev2_maxout0_maxout_b_read_readvariableop%savev2_bn1_norm_w_read_readvariableop%savev2_bn1_norm_b_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop%savev2_bn3_norm_w_read_readvariableop%savev2_bn3_norm_b_read_readvariableop(savev2_dense4_kernel_read_readvariableop&savev2_dense4_bias_read_readvariableop%savev2_bn5_norm_w_read_readvariableop%savev2_bn5_norm_b_read_readvariableop(savev2_dense6_kernel_read_readvariableop&savev2_dense6_bias_read_readvariableop%savev2_bn7_norm_w_read_readvariableop%savev2_bn7_norm_b_read_readvariableop(savev2_dense8_kernel_read_readvariableop&savev2_dense8_bias_read_readvariableop%savev2_bn9_norm_w_read_readvariableop%savev2_bn9_norm_b_read_readvariableop,savev2_maxout10_maxout_w_read_readvariableop,savev2_maxout10_maxout_b_read_readvariableop&savev2_bn11_norm_w_read_readvariableop&savev2_bn11_norm_b_read_readvariableop)savev2_dense12_kernel_read_readvariableop'savev2_dense12_bias_read_readvariableop&savev2_bn13_norm_w_read_readvariableop&savev2_bn13_norm_b_read_readvariableop)savev2_dense14_kernel_read_readvariableop'savev2_dense14_bias_read_readvariableop&savev2_bn15_norm_w_read_readvariableop&savev2_bn15_norm_b_read_readvariableop)savev2_dense16_kernel_read_readvariableop'savev2_dense16_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :	)?:?:HH:H:H9:9:99:9:9<:<:<<:<:<0:0:00:0:0$:$:$$:$:	$?:?::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	)?:!

_output_shapes	
:?:$ 

_output_shapes

:HH: 

_output_shapes
:H:$ 

_output_shapes

:H9: 

_output_shapes
:9:$ 

_output_shapes

:99: 

_output_shapes
:9:$	 

_output_shapes

:9<: 


_output_shapes
:<:$ 

_output_shapes

:<<: 

_output_shapes
:<:$ 

_output_shapes

:<0: 

_output_shapes
:0:$ 

_output_shapes

:00: 

_output_shapes
:0:$ 

_output_shapes

:0$: 

_output_shapes
:$:$ 

_output_shapes

:$$: 

_output_shapes
:$:%!

_output_shapes
:	$?:!

_output_shapes	
:?:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::#

_output_shapes
: 
?
?
=__inference_BN5_layer_call_and_return_conditional_losses_2215

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:<*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
@__inference_dense6_layer_call_and_return_conditional_losses_2235

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
x
#__inference_BN13_layer_call_fn_2395

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN13_layer_call_and_return_conditional_losses_10702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
<__inference_BN7_layer_call_and_return_conditional_losses_896

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:0*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:00*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?	
?
@__inference_dense4_layer_call_and_return_conditional_losses_2196

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:9<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?P
?	
?__inference_model_layer_call_and_return_conditional_losses_1348

inputs
maxout0_1262
maxout0_1264
bn1_1267
bn1_1269
dense2_1272
dense2_1274
bn3_1277
bn3_1279
dense4_1282
dense4_1284
bn5_1287
bn5_1289
dense6_1292
dense6_1294
bn7_1297
bn7_1299
dense8_1302
dense8_1304
bn9_1307
bn9_1309
maxout10_1312
maxout10_1314
	bn11_1317
	bn11_1319
dense12_1322
dense12_1324
	bn13_1327
	bn13_1329
dense14_1332
dense14_1334
	bn15_1337
	bn15_1339
dense16_1342
dense16_1344
identity??BN1/StatefulPartitionedCall?BN11/StatefulPartitionedCall?BN13/StatefulPartitionedCall?BN15/StatefulPartitionedCall?BN3/StatefulPartitionedCall?BN5/StatefulPartitionedCall?BN7/StatefulPartitionedCall?BN9/StatefulPartitionedCall?dense12/StatefulPartitionedCall?dense14/StatefulPartitionedCall?dense16/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense4/StatefulPartitionedCall?dense6/StatefulPartitionedCall?dense8/StatefulPartitionedCall?maxout0/StatefulPartitionedCall? maxout10/StatefulPartitionedCall?
maxout0/StatefulPartitionedCallStatefulPartitionedCallinputsmaxout0_1262maxout0_1264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_maxout0_layer_call_and_return_conditional_losses_7112!
maxout0/StatefulPartitionedCall?
BN1/StatefulPartitionedCallStatefulPartitionedCall(maxout0/StatefulPartitionedCall:output:0bn1_1267bn1_1269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN1_layer_call_and_return_conditional_losses_7372
BN1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall$BN1/StatefulPartitionedCall:output:0dense2_1272dense2_1274*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense2_layer_call_and_return_conditional_losses_7642 
dense2/StatefulPartitionedCall?
BN3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0bn3_1277bn3_1279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN3_layer_call_and_return_conditional_losses_7902
BN3/StatefulPartitionedCall?
dense4/StatefulPartitionedCallStatefulPartitionedCall$BN3/StatefulPartitionedCall:output:0dense4_1282dense4_1284*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense4_layer_call_and_return_conditional_losses_8172 
dense4/StatefulPartitionedCall?
BN5/StatefulPartitionedCallStatefulPartitionedCall'dense4/StatefulPartitionedCall:output:0bn5_1287bn5_1289*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN5_layer_call_and_return_conditional_losses_8432
BN5/StatefulPartitionedCall?
dense6/StatefulPartitionedCallStatefulPartitionedCall$BN5/StatefulPartitionedCall:output:0dense6_1292dense6_1294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense6_layer_call_and_return_conditional_losses_8702 
dense6/StatefulPartitionedCall?
BN7/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0bn7_1297bn7_1299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN7_layer_call_and_return_conditional_losses_8962
BN7/StatefulPartitionedCall?
dense8/StatefulPartitionedCallStatefulPartitionedCall$BN7/StatefulPartitionedCall:output:0dense8_1302dense8_1304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense8_layer_call_and_return_conditional_losses_9232 
dense8/StatefulPartitionedCall?
BN9/StatefulPartitionedCallStatefulPartitionedCall'dense8/StatefulPartitionedCall:output:0bn9_1307bn9_1309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN9_layer_call_and_return_conditional_losses_9492
BN9/StatefulPartitionedCall?
 maxout10/StatefulPartitionedCallStatefulPartitionedCall$BN9/StatefulPartitionedCall:output:0maxout10_1312maxout10_1314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_maxout10_layer_call_and_return_conditional_losses_9912"
 maxout10/StatefulPartitionedCall?
BN11/StatefulPartitionedCallStatefulPartitionedCall)maxout10/StatefulPartitionedCall:output:0	bn11_1317	bn11_1319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN11_layer_call_and_return_conditional_losses_10172
BN11/StatefulPartitionedCall?
dense12/StatefulPartitionedCallStatefulPartitionedCall%BN11/StatefulPartitionedCall:output:0dense12_1322dense12_1324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense12_layer_call_and_return_conditional_losses_10442!
dense12/StatefulPartitionedCall?
BN13/StatefulPartitionedCallStatefulPartitionedCall(dense12/StatefulPartitionedCall:output:0	bn13_1327	bn13_1329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN13_layer_call_and_return_conditional_losses_10702
BN13/StatefulPartitionedCall?
dense14/StatefulPartitionedCallStatefulPartitionedCall%BN13/StatefulPartitionedCall:output:0dense14_1332dense14_1334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense14_layer_call_and_return_conditional_losses_10972!
dense14/StatefulPartitionedCall?
BN15/StatefulPartitionedCallStatefulPartitionedCall(dense14/StatefulPartitionedCall:output:0	bn15_1337	bn15_1339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN15_layer_call_and_return_conditional_losses_11232
BN15/StatefulPartitionedCall?
dense16/StatefulPartitionedCallStatefulPartitionedCall%BN15/StatefulPartitionedCall:output:0dense16_1342dense16_1344*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense16_layer_call_and_return_conditional_losses_11502!
dense16/StatefulPartitionedCall?
IdentityIdentity(dense16/StatefulPartitionedCall:output:0^BN1/StatefulPartitionedCall^BN11/StatefulPartitionedCall^BN13/StatefulPartitionedCall^BN15/StatefulPartitionedCall^BN3/StatefulPartitionedCall^BN5/StatefulPartitionedCall^BN7/StatefulPartitionedCall^BN9/StatefulPartitionedCall ^dense12/StatefulPartitionedCall ^dense14/StatefulPartitionedCall ^dense16/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense4/StatefulPartitionedCall^dense6/StatefulPartitionedCall^dense8/StatefulPartitionedCall ^maxout0/StatefulPartitionedCall!^maxout10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::2:
BN1/StatefulPartitionedCallBN1/StatefulPartitionedCall2<
BN11/StatefulPartitionedCallBN11/StatefulPartitionedCall2<
BN13/StatefulPartitionedCallBN13/StatefulPartitionedCall2<
BN15/StatefulPartitionedCallBN15/StatefulPartitionedCall2:
BN3/StatefulPartitionedCallBN3/StatefulPartitionedCall2:
BN5/StatefulPartitionedCallBN5/StatefulPartitionedCall2:
BN7/StatefulPartitionedCallBN7/StatefulPartitionedCall2:
BN9/StatefulPartitionedCallBN9/StatefulPartitionedCall2B
dense12/StatefulPartitionedCalldense12/StatefulPartitionedCall2B
dense14/StatefulPartitionedCalldense14/StatefulPartitionedCall2B
dense16/StatefulPartitionedCalldense16/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall2@
dense8/StatefulPartitionedCalldense8/StatefulPartitionedCall2B
maxout0/StatefulPartitionedCallmaxout0/StatefulPartitionedCall2D
 maxout10/StatefulPartitionedCall maxout10/StatefulPartitionedCall:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
z
%__inference_dense2_layer_call_fn_2166

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense2_layer_call_and_return_conditional_losses_7642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????92

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????H::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
{
&__inference_maxout0_layer_call_fn_2127

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_maxout0_layer_call_and_return_conditional_losses_7112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????)::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
|
'__inference_maxout10_layer_call_fn_2337

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_maxout10_layer_call_and_return_conditional_losses_9912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs
?
?
>__inference_BN15_layer_call_and_return_conditional_losses_2425

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?P
?	
?__inference_model_layer_call_and_return_conditional_losses_1510

inputs
maxout0_1424
maxout0_1426
bn1_1429
bn1_1431
dense2_1434
dense2_1436
bn3_1439
bn3_1441
dense4_1444
dense4_1446
bn5_1449
bn5_1451
dense6_1454
dense6_1456
bn7_1459
bn7_1461
dense8_1464
dense8_1466
bn9_1469
bn9_1471
maxout10_1474
maxout10_1476
	bn11_1479
	bn11_1481
dense12_1484
dense12_1486
	bn13_1489
	bn13_1491
dense14_1494
dense14_1496
	bn15_1499
	bn15_1501
dense16_1504
dense16_1506
identity??BN1/StatefulPartitionedCall?BN11/StatefulPartitionedCall?BN13/StatefulPartitionedCall?BN15/StatefulPartitionedCall?BN3/StatefulPartitionedCall?BN5/StatefulPartitionedCall?BN7/StatefulPartitionedCall?BN9/StatefulPartitionedCall?dense12/StatefulPartitionedCall?dense14/StatefulPartitionedCall?dense16/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense4/StatefulPartitionedCall?dense6/StatefulPartitionedCall?dense8/StatefulPartitionedCall?maxout0/StatefulPartitionedCall? maxout10/StatefulPartitionedCall?
maxout0/StatefulPartitionedCallStatefulPartitionedCallinputsmaxout0_1424maxout0_1426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_maxout0_layer_call_and_return_conditional_losses_7112!
maxout0/StatefulPartitionedCall?
BN1/StatefulPartitionedCallStatefulPartitionedCall(maxout0/StatefulPartitionedCall:output:0bn1_1429bn1_1431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN1_layer_call_and_return_conditional_losses_7372
BN1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall$BN1/StatefulPartitionedCall:output:0dense2_1434dense2_1436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense2_layer_call_and_return_conditional_losses_7642 
dense2/StatefulPartitionedCall?
BN3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0bn3_1439bn3_1441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN3_layer_call_and_return_conditional_losses_7902
BN3/StatefulPartitionedCall?
dense4/StatefulPartitionedCallStatefulPartitionedCall$BN3/StatefulPartitionedCall:output:0dense4_1444dense4_1446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense4_layer_call_and_return_conditional_losses_8172 
dense4/StatefulPartitionedCall?
BN5/StatefulPartitionedCallStatefulPartitionedCall'dense4/StatefulPartitionedCall:output:0bn5_1449bn5_1451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN5_layer_call_and_return_conditional_losses_8432
BN5/StatefulPartitionedCall?
dense6/StatefulPartitionedCallStatefulPartitionedCall$BN5/StatefulPartitionedCall:output:0dense6_1454dense6_1456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense6_layer_call_and_return_conditional_losses_8702 
dense6/StatefulPartitionedCall?
BN7/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0bn7_1459bn7_1461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN7_layer_call_and_return_conditional_losses_8962
BN7/StatefulPartitionedCall?
dense8/StatefulPartitionedCallStatefulPartitionedCall$BN7/StatefulPartitionedCall:output:0dense8_1464dense8_1466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense8_layer_call_and_return_conditional_losses_9232 
dense8/StatefulPartitionedCall?
BN9/StatefulPartitionedCallStatefulPartitionedCall'dense8/StatefulPartitionedCall:output:0bn9_1469bn9_1471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN9_layer_call_and_return_conditional_losses_9492
BN9/StatefulPartitionedCall?
 maxout10/StatefulPartitionedCallStatefulPartitionedCall$BN9/StatefulPartitionedCall:output:0maxout10_1474maxout10_1476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_maxout10_layer_call_and_return_conditional_losses_9912"
 maxout10/StatefulPartitionedCall?
BN11/StatefulPartitionedCallStatefulPartitionedCall)maxout10/StatefulPartitionedCall:output:0	bn11_1479	bn11_1481*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN11_layer_call_and_return_conditional_losses_10172
BN11/StatefulPartitionedCall?
dense12/StatefulPartitionedCallStatefulPartitionedCall%BN11/StatefulPartitionedCall:output:0dense12_1484dense12_1486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense12_layer_call_and_return_conditional_losses_10442!
dense12/StatefulPartitionedCall?
BN13/StatefulPartitionedCallStatefulPartitionedCall(dense12/StatefulPartitionedCall:output:0	bn13_1489	bn13_1491*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN13_layer_call_and_return_conditional_losses_10702
BN13/StatefulPartitionedCall?
dense14/StatefulPartitionedCallStatefulPartitionedCall%BN13/StatefulPartitionedCall:output:0dense14_1494dense14_1496*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense14_layer_call_and_return_conditional_losses_10972!
dense14/StatefulPartitionedCall?
BN15/StatefulPartitionedCallStatefulPartitionedCall(dense14/StatefulPartitionedCall:output:0	bn15_1499	bn15_1501*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN15_layer_call_and_return_conditional_losses_11232
BN15/StatefulPartitionedCall?
dense16/StatefulPartitionedCallStatefulPartitionedCall%BN15/StatefulPartitionedCall:output:0dense16_1504dense16_1506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense16_layer_call_and_return_conditional_losses_11502!
dense16/StatefulPartitionedCall?
IdentityIdentity(dense16/StatefulPartitionedCall:output:0^BN1/StatefulPartitionedCall^BN11/StatefulPartitionedCall^BN13/StatefulPartitionedCall^BN15/StatefulPartitionedCall^BN3/StatefulPartitionedCall^BN5/StatefulPartitionedCall^BN7/StatefulPartitionedCall^BN9/StatefulPartitionedCall ^dense12/StatefulPartitionedCall ^dense14/StatefulPartitionedCall ^dense16/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense4/StatefulPartitionedCall^dense6/StatefulPartitionedCall^dense8/StatefulPartitionedCall ^maxout0/StatefulPartitionedCall!^maxout10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::2:
BN1/StatefulPartitionedCallBN1/StatefulPartitionedCall2<
BN11/StatefulPartitionedCallBN11/StatefulPartitionedCall2<
BN13/StatefulPartitionedCallBN13/StatefulPartitionedCall2<
BN15/StatefulPartitionedCallBN15/StatefulPartitionedCall2:
BN3/StatefulPartitionedCallBN3/StatefulPartitionedCall2:
BN5/StatefulPartitionedCallBN5/StatefulPartitionedCall2:
BN7/StatefulPartitionedCallBN7/StatefulPartitionedCall2:
BN9/StatefulPartitionedCallBN9/StatefulPartitionedCall2B
dense12/StatefulPartitionedCalldense12/StatefulPartitionedCall2B
dense14/StatefulPartitionedCalldense14/StatefulPartitionedCall2B
dense16/StatefulPartitionedCalldense16/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall2@
dense8/StatefulPartitionedCalldense8/StatefulPartitionedCall2B
maxout0/StatefulPartitionedCallmaxout0/StatefulPartitionedCall2D
 maxout10/StatefulPartitionedCall maxout10/StatefulPartitionedCall:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
z
%__inference_dense4_layer_call_fn_2205

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense4_layer_call_and_return_conditional_losses_8172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_2019

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_13482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
w
"__inference_BN5_layer_call_fn_2224

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN5_layer_call_and_return_conditional_losses_8432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
>__inference_BN13_layer_call_and_return_conditional_losses_1070

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_1946

inputs*
&maxout0_matmul_readvariableop_resource'
#maxout0_add_readvariableop_resource#
bn1_add_readvariableop_resource&
"bn1_matmul_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource#
bn3_add_readvariableop_resource&
"bn3_matmul_readvariableop_resource)
%dense4_matmul_readvariableop_resource*
&dense4_biasadd_readvariableop_resource#
bn5_add_readvariableop_resource&
"bn5_matmul_readvariableop_resource)
%dense6_matmul_readvariableop_resource*
&dense6_biasadd_readvariableop_resource#
bn7_add_readvariableop_resource&
"bn7_matmul_readvariableop_resource)
%dense8_matmul_readvariableop_resource*
&dense8_biasadd_readvariableop_resource#
bn9_add_readvariableop_resource&
"bn9_matmul_readvariableop_resource+
'maxout10_matmul_readvariableop_resource(
$maxout10_add_readvariableop_resource$
 bn11_add_readvariableop_resource'
#bn11_matmul_readvariableop_resource*
&dense12_matmul_readvariableop_resource+
'dense12_biasadd_readvariableop_resource$
 bn13_add_readvariableop_resource'
#bn13_matmul_readvariableop_resource*
&dense14_matmul_readvariableop_resource+
'dense14_biasadd_readvariableop_resource$
 bn15_add_readvariableop_resource'
#bn15_matmul_readvariableop_resource*
&dense16_matmul_readvariableop_resource+
'dense16_biasadd_readvariableop_resource
identity??BN1/MatMul/ReadVariableOp?BN1/add/ReadVariableOp?BN11/MatMul/ReadVariableOp?BN11/add/ReadVariableOp?BN13/MatMul/ReadVariableOp?BN13/add/ReadVariableOp?BN15/MatMul/ReadVariableOp?BN15/add/ReadVariableOp?BN3/MatMul/ReadVariableOp?BN3/add/ReadVariableOp?BN5/MatMul/ReadVariableOp?BN5/add/ReadVariableOp?BN7/MatMul/ReadVariableOp?BN7/add/ReadVariableOp?BN9/MatMul/ReadVariableOp?BN9/add/ReadVariableOp?dense12/BiasAdd/ReadVariableOp?dense12/MatMul/ReadVariableOp?dense14/BiasAdd/ReadVariableOp?dense14/MatMul/ReadVariableOp?dense16/BiasAdd/ReadVariableOp?dense16/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?dense4/BiasAdd/ReadVariableOp?dense4/MatMul/ReadVariableOp?dense6/BiasAdd/ReadVariableOp?dense6/MatMul/ReadVariableOp?dense8/BiasAdd/ReadVariableOp?dense8/MatMul/ReadVariableOp?maxout0/MatMul/ReadVariableOp?maxout0/add/ReadVariableOp?maxout10/MatMul/ReadVariableOp?maxout10/add/ReadVariableOp?
maxout0/MatMul/ReadVariableOpReadVariableOp&maxout0_matmul_readvariableop_resource*
_output_shapes
:	)?*
dtype02
maxout0/MatMul/ReadVariableOp?
maxout0/MatMulMatMulinputs%maxout0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
maxout0/MatMul?
maxout0/add/ReadVariableOpReadVariableOp#maxout0_add_readvariableop_resource*
_output_shapes	
:?*
dtype02
maxout0/add/ReadVariableOp?
maxout0/addAddV2maxout0/MatMul:product:0"maxout0/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
maxout0/add?
maxout0/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   H   2
maxout0/Reshape/shape?
maxout0/ReshapeReshapemaxout0/add:z:0maxout0/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????H2
maxout0/Reshapet
maxout0/maxout/ShapeShapemaxout0/Reshape:output:0*
T0*
_output_shapes
:2
maxout0/maxout/Shape?
"maxout0/maxout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"maxout0/maxout/strided_slice/stack?
$maxout0/maxout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$maxout0/maxout/strided_slice/stack_1?
$maxout0/maxout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$maxout0/maxout/strided_slice/stack_2?
maxout0/maxout/strided_sliceStridedSlicemaxout0/maxout/Shape:output:0+maxout0/maxout/strided_slice/stack:output:0-maxout0/maxout/strided_slice/stack_1:output:0-maxout0/maxout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
maxout0/maxout/strided_slice?
maxout0/maxout/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
maxout0/maxout/Reshape/shape/1?
maxout0/maxout/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
maxout0/maxout/Reshape/shape/2?
maxout0/maxout/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :H2 
maxout0/maxout/Reshape/shape/3?
maxout0/maxout/Reshape/shapePack%maxout0/maxout/strided_slice:output:0'maxout0/maxout/Reshape/shape/1:output:0'maxout0/maxout/Reshape/shape/2:output:0'maxout0/maxout/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
maxout0/maxout/Reshape/shape?
maxout0/maxout/ReshapeReshapemaxout0/Reshape:output:0%maxout0/maxout/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????H2
maxout0/maxout/Reshape?
$maxout0/maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2&
$maxout0/maxout/Max/reduction_indices?
maxout0/maxout/MaxMaxmaxout0/maxout/Reshape:output:0-maxout0/maxout/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????H2
maxout0/maxout/Max?
maxout0/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????H   2
maxout0/Reshape_1/shape?
maxout0/Reshape_1Reshapemaxout0/maxout/Max:output:0 maxout0/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????H2
maxout0/Reshape_1?
BN1/add/ReadVariableOpReadVariableOpbn1_add_readvariableop_resource*
_output_shapes
:H*
dtype02
BN1/add/ReadVariableOp?
BN1/addAddV2maxout0/Reshape_1:output:0BN1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BN1/add?
BN1/MatMul/ReadVariableOpReadVariableOp"bn1_matmul_readvariableop_resource*
_output_shapes

:HH*
dtype02
BN1/MatMul/ReadVariableOp?

BN1/MatMulMatMulmaxout0/Reshape_1:output:0!BN1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2

BN1/MatMul?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:H9*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMulBN1/MatMul:product:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????92
dense2/Relu?
BN3/add/ReadVariableOpReadVariableOpbn3_add_readvariableop_resource*
_output_shapes
:9*
dtype02
BN3/add/ReadVariableOp?
BN3/addAddV2dense2/Relu:activations:0BN3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92	
BN3/add?
BN3/MatMul/ReadVariableOpReadVariableOp"bn3_matmul_readvariableop_resource*
_output_shapes

:99*
dtype02
BN3/MatMul/ReadVariableOp?

BN3/MatMulMatMuldense2/Relu:activations:0!BN3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92

BN3/MatMul?
dense4/MatMul/ReadVariableOpReadVariableOp%dense4_matmul_readvariableop_resource*
_output_shapes

:9<*
dtype02
dense4/MatMul/ReadVariableOp?
dense4/MatMulMatMulBN3/MatMul:product:0$dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dense4/MatMul?
dense4/BiasAdd/ReadVariableOpReadVariableOp&dense4_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
dense4/BiasAdd/ReadVariableOp?
dense4/BiasAddBiasAdddense4/MatMul:product:0%dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dense4/BiasAddm
dense4/ReluReludense4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dense4/Relu?
BN5/add/ReadVariableOpReadVariableOpbn5_add_readvariableop_resource*
_output_shapes
:<*
dtype02
BN5/add/ReadVariableOp?
BN5/addAddV2dense4/Relu:activations:0BN5/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BN5/add?
BN5/MatMul/ReadVariableOpReadVariableOp"bn5_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
BN5/MatMul/ReadVariableOp?

BN5/MatMulMatMuldense4/Relu:activations:0!BN5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2

BN5/MatMul?
dense6/MatMul/ReadVariableOpReadVariableOp%dense6_matmul_readvariableop_resource*
_output_shapes

:<0*
dtype02
dense6/MatMul/ReadVariableOp?
dense6/MatMulMatMulBN5/MatMul:product:0$dense6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense6/MatMul?
dense6/BiasAdd/ReadVariableOpReadVariableOp&dense6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
dense6/BiasAdd/ReadVariableOp?
dense6/BiasAddBiasAdddense6/MatMul:product:0%dense6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense6/BiasAddm
dense6/ReluReludense6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
dense6/Relu?
BN7/add/ReadVariableOpReadVariableOpbn7_add_readvariableop_resource*
_output_shapes
:0*
dtype02
BN7/add/ReadVariableOp?
BN7/addAddV2dense6/Relu:activations:0BN7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BN7/add?
BN7/MatMul/ReadVariableOpReadVariableOp"bn7_matmul_readvariableop_resource*
_output_shapes

:00*
dtype02
BN7/MatMul/ReadVariableOp?

BN7/MatMulMatMuldense6/Relu:activations:0!BN7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02

BN7/MatMul?
dense8/MatMul/ReadVariableOpReadVariableOp%dense8_matmul_readvariableop_resource*
_output_shapes

:0$*
dtype02
dense8/MatMul/ReadVariableOp?
dense8/MatMulMatMulBN7/MatMul:product:0$dense8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense8/MatMul?
dense8/BiasAdd/ReadVariableOpReadVariableOp&dense8_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
dense8/BiasAdd/ReadVariableOp?
dense8/BiasAddBiasAdddense8/MatMul:product:0%dense8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense8/BiasAddm
dense8/ReluReludense8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
dense8/Relu?
BN9/add/ReadVariableOpReadVariableOpbn9_add_readvariableop_resource*
_output_shapes
:$*
dtype02
BN9/add/ReadVariableOp?
BN9/addAddV2dense8/Relu:activations:0BN9/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2	
BN9/add?
BN9/MatMul/ReadVariableOpReadVariableOp"bn9_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
BN9/MatMul/ReadVariableOp?

BN9/MatMulMatMuldense8/Relu:activations:0!BN9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2

BN9/MatMul?
maxout10/MatMul/ReadVariableOpReadVariableOp'maxout10_matmul_readvariableop_resource*
_output_shapes
:	$?*
dtype02 
maxout10/MatMul/ReadVariableOp?
maxout10/MatMulMatMulBN9/MatMul:product:0&maxout10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
maxout10/MatMul?
maxout10/add/ReadVariableOpReadVariableOp$maxout10_add_readvariableop_resource*
_output_shapes	
:?*
dtype02
maxout10/add/ReadVariableOp?
maxout10/addAddV2maxout10/MatMul:product:0#maxout10/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
maxout10/add?
maxout10/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
maxout10/Reshape/shape?
maxout10/ReshapeReshapemaxout10/add:z:0maxout10/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
maxout10/Reshape{
maxout10/maxout_1/ShapeShapemaxout10/Reshape:output:0*
T0*
_output_shapes
:2
maxout10/maxout_1/Shape?
%maxout10/maxout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%maxout10/maxout_1/strided_slice/stack?
'maxout10/maxout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'maxout10/maxout_1/strided_slice/stack_1?
'maxout10/maxout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'maxout10/maxout_1/strided_slice/stack_2?
maxout10/maxout_1/strided_sliceStridedSlice maxout10/maxout_1/Shape:output:0.maxout10/maxout_1/strided_slice/stack:output:00maxout10/maxout_1/strided_slice/stack_1:output:00maxout10/maxout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
maxout10/maxout_1/strided_slice?
!maxout10/maxout_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!maxout10/maxout_1/Reshape/shape/1?
!maxout10/maxout_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!maxout10/maxout_1/Reshape/shape/2?
!maxout10/maxout_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!maxout10/maxout_1/Reshape/shape/3?
maxout10/maxout_1/Reshape/shapePack(maxout10/maxout_1/strided_slice:output:0*maxout10/maxout_1/Reshape/shape/1:output:0*maxout10/maxout_1/Reshape/shape/2:output:0*maxout10/maxout_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
maxout10/maxout_1/Reshape/shape?
maxout10/maxout_1/ReshapeReshapemaxout10/Reshape:output:0(maxout10/maxout_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
maxout10/maxout_1/Reshape?
'maxout10/maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2)
'maxout10/maxout_1/Max/reduction_indices?
maxout10/maxout_1/MaxMax"maxout10/maxout_1/Reshape:output:00maxout10/maxout_1/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2
maxout10/maxout_1/Max?
maxout10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
maxout10/Reshape_1/shape?
maxout10/Reshape_1Reshapemaxout10/maxout_1/Max:output:0!maxout10/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
maxout10/Reshape_1?
BN11/add/ReadVariableOpReadVariableOp bn11_add_readvariableop_resource*
_output_shapes
:*
dtype02
BN11/add/ReadVariableOp?
BN11/addAddV2maxout10/Reshape_1:output:0BN11/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

BN11/add?
BN11/MatMul/ReadVariableOpReadVariableOp#bn11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
BN11/MatMul/ReadVariableOp?
BN11/MatMulMatMulmaxout10/Reshape_1:output:0"BN11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
BN11/MatMul?
dense12/MatMul/ReadVariableOpReadVariableOp&dense12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense12/MatMul/ReadVariableOp?
dense12/MatMulMatMulBN11/MatMul:product:0%dense12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense12/MatMul?
dense12/BiasAdd/ReadVariableOpReadVariableOp'dense12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense12/BiasAdd/ReadVariableOp?
dense12/BiasAddBiasAdddense12/MatMul:product:0&dense12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense12/BiasAddp
dense12/ReluReludense12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense12/Relu?
BN13/add/ReadVariableOpReadVariableOp bn13_add_readvariableop_resource*
_output_shapes
:*
dtype02
BN13/add/ReadVariableOp?
BN13/addAddV2dense12/Relu:activations:0BN13/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

BN13/add?
BN13/MatMul/ReadVariableOpReadVariableOp#bn13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
BN13/MatMul/ReadVariableOp?
BN13/MatMulMatMuldense12/Relu:activations:0"BN13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
BN13/MatMul?
dense14/MatMul/ReadVariableOpReadVariableOp&dense14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense14/MatMul/ReadVariableOp?
dense14/MatMulMatMulBN13/MatMul:product:0%dense14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense14/MatMul?
dense14/BiasAdd/ReadVariableOpReadVariableOp'dense14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense14/BiasAdd/ReadVariableOp?
dense14/BiasAddBiasAdddense14/MatMul:product:0&dense14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense14/BiasAddp
dense14/ReluReludense14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense14/Relu?
BN15/add/ReadVariableOpReadVariableOp bn15_add_readvariableop_resource*
_output_shapes
:*
dtype02
BN15/add/ReadVariableOp?
BN15/addAddV2dense14/Relu:activations:0BN15/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

BN15/add?
BN15/MatMul/ReadVariableOpReadVariableOp#bn15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
BN15/MatMul/ReadVariableOp?
BN15/MatMulMatMuldense14/Relu:activations:0"BN15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
BN15/MatMul?
dense16/MatMul/ReadVariableOpReadVariableOp&dense16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense16/MatMul/ReadVariableOp?
dense16/MatMulMatMulBN15/MatMul:product:0%dense16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense16/MatMul?
dense16/BiasAdd/ReadVariableOpReadVariableOp'dense16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense16/BiasAdd/ReadVariableOp?
dense16/BiasAddBiasAdddense16/MatMul:product:0&dense16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense16/BiasAddy
dense16/SoftmaxSoftmaxdense16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense16/Softmax?
IdentityIdentitydense16/Softmax:softmax:0^BN1/MatMul/ReadVariableOp^BN1/add/ReadVariableOp^BN11/MatMul/ReadVariableOp^BN11/add/ReadVariableOp^BN13/MatMul/ReadVariableOp^BN13/add/ReadVariableOp^BN15/MatMul/ReadVariableOp^BN15/add/ReadVariableOp^BN3/MatMul/ReadVariableOp^BN3/add/ReadVariableOp^BN5/MatMul/ReadVariableOp^BN5/add/ReadVariableOp^BN7/MatMul/ReadVariableOp^BN7/add/ReadVariableOp^BN9/MatMul/ReadVariableOp^BN9/add/ReadVariableOp^dense12/BiasAdd/ReadVariableOp^dense12/MatMul/ReadVariableOp^dense14/BiasAdd/ReadVariableOp^dense14/MatMul/ReadVariableOp^dense16/BiasAdd/ReadVariableOp^dense16/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^dense4/BiasAdd/ReadVariableOp^dense4/MatMul/ReadVariableOp^dense6/BiasAdd/ReadVariableOp^dense6/MatMul/ReadVariableOp^dense8/BiasAdd/ReadVariableOp^dense8/MatMul/ReadVariableOp^maxout0/MatMul/ReadVariableOp^maxout0/add/ReadVariableOp^maxout10/MatMul/ReadVariableOp^maxout10/add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::26
BN1/MatMul/ReadVariableOpBN1/MatMul/ReadVariableOp20
BN1/add/ReadVariableOpBN1/add/ReadVariableOp28
BN11/MatMul/ReadVariableOpBN11/MatMul/ReadVariableOp22
BN11/add/ReadVariableOpBN11/add/ReadVariableOp28
BN13/MatMul/ReadVariableOpBN13/MatMul/ReadVariableOp22
BN13/add/ReadVariableOpBN13/add/ReadVariableOp28
BN15/MatMul/ReadVariableOpBN15/MatMul/ReadVariableOp22
BN15/add/ReadVariableOpBN15/add/ReadVariableOp26
BN3/MatMul/ReadVariableOpBN3/MatMul/ReadVariableOp20
BN3/add/ReadVariableOpBN3/add/ReadVariableOp26
BN5/MatMul/ReadVariableOpBN5/MatMul/ReadVariableOp20
BN5/add/ReadVariableOpBN5/add/ReadVariableOp26
BN7/MatMul/ReadVariableOpBN7/MatMul/ReadVariableOp20
BN7/add/ReadVariableOpBN7/add/ReadVariableOp26
BN9/MatMul/ReadVariableOpBN9/MatMul/ReadVariableOp20
BN9/add/ReadVariableOpBN9/add/ReadVariableOp2@
dense12/BiasAdd/ReadVariableOpdense12/BiasAdd/ReadVariableOp2>
dense12/MatMul/ReadVariableOpdense12/MatMul/ReadVariableOp2@
dense14/BiasAdd/ReadVariableOpdense14/BiasAdd/ReadVariableOp2>
dense14/MatMul/ReadVariableOpdense14/MatMul/ReadVariableOp2@
dense16/BiasAdd/ReadVariableOpdense16/BiasAdd/ReadVariableOp2>
dense16/MatMul/ReadVariableOpdense16/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2>
dense4/BiasAdd/ReadVariableOpdense4/BiasAdd/ReadVariableOp2<
dense4/MatMul/ReadVariableOpdense4/MatMul/ReadVariableOp2>
dense6/BiasAdd/ReadVariableOpdense6/BiasAdd/ReadVariableOp2<
dense6/MatMul/ReadVariableOpdense6/MatMul/ReadVariableOp2>
dense8/BiasAdd/ReadVariableOpdense8/BiasAdd/ReadVariableOp2<
dense8/MatMul/ReadVariableOpdense8/MatMul/ReadVariableOp2>
maxout0/MatMul/ReadVariableOpmaxout0/MatMul/ReadVariableOp28
maxout0/add/ReadVariableOpmaxout0/add/ReadVariableOp2@
maxout10/MatMul/ReadVariableOpmaxout10/MatMul/ReadVariableOp2:
maxout10/add/ReadVariableOpmaxout10/add/ReadVariableOp:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
?
<__inference_BN3_layer_call_and_return_conditional_losses_790

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:9*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:99*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????92

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
z
%__inference_dense6_layer_call_fn_2244

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense6_layer_call_and_return_conditional_losses_8702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_681	
input0
,model_maxout0_matmul_readvariableop_resource-
)model_maxout0_add_readvariableop_resource)
%model_bn1_add_readvariableop_resource,
(model_bn1_matmul_readvariableop_resource/
+model_dense2_matmul_readvariableop_resource0
,model_dense2_biasadd_readvariableop_resource)
%model_bn3_add_readvariableop_resource,
(model_bn3_matmul_readvariableop_resource/
+model_dense4_matmul_readvariableop_resource0
,model_dense4_biasadd_readvariableop_resource)
%model_bn5_add_readvariableop_resource,
(model_bn5_matmul_readvariableop_resource/
+model_dense6_matmul_readvariableop_resource0
,model_dense6_biasadd_readvariableop_resource)
%model_bn7_add_readvariableop_resource,
(model_bn7_matmul_readvariableop_resource/
+model_dense8_matmul_readvariableop_resource0
,model_dense8_biasadd_readvariableop_resource)
%model_bn9_add_readvariableop_resource,
(model_bn9_matmul_readvariableop_resource1
-model_maxout10_matmul_readvariableop_resource.
*model_maxout10_add_readvariableop_resource*
&model_bn11_add_readvariableop_resource-
)model_bn11_matmul_readvariableop_resource0
,model_dense12_matmul_readvariableop_resource1
-model_dense12_biasadd_readvariableop_resource*
&model_bn13_add_readvariableop_resource-
)model_bn13_matmul_readvariableop_resource0
,model_dense14_matmul_readvariableop_resource1
-model_dense14_biasadd_readvariableop_resource*
&model_bn15_add_readvariableop_resource-
)model_bn15_matmul_readvariableop_resource0
,model_dense16_matmul_readvariableop_resource1
-model_dense16_biasadd_readvariableop_resource
identity??model/BN1/MatMul/ReadVariableOp?model/BN1/add/ReadVariableOp? model/BN11/MatMul/ReadVariableOp?model/BN11/add/ReadVariableOp? model/BN13/MatMul/ReadVariableOp?model/BN13/add/ReadVariableOp? model/BN15/MatMul/ReadVariableOp?model/BN15/add/ReadVariableOp?model/BN3/MatMul/ReadVariableOp?model/BN3/add/ReadVariableOp?model/BN5/MatMul/ReadVariableOp?model/BN5/add/ReadVariableOp?model/BN7/MatMul/ReadVariableOp?model/BN7/add/ReadVariableOp?model/BN9/MatMul/ReadVariableOp?model/BN9/add/ReadVariableOp?$model/dense12/BiasAdd/ReadVariableOp?#model/dense12/MatMul/ReadVariableOp?$model/dense14/BiasAdd/ReadVariableOp?#model/dense14/MatMul/ReadVariableOp?$model/dense16/BiasAdd/ReadVariableOp?#model/dense16/MatMul/ReadVariableOp?#model/dense2/BiasAdd/ReadVariableOp?"model/dense2/MatMul/ReadVariableOp?#model/dense4/BiasAdd/ReadVariableOp?"model/dense4/MatMul/ReadVariableOp?#model/dense6/BiasAdd/ReadVariableOp?"model/dense6/MatMul/ReadVariableOp?#model/dense8/BiasAdd/ReadVariableOp?"model/dense8/MatMul/ReadVariableOp?#model/maxout0/MatMul/ReadVariableOp? model/maxout0/add/ReadVariableOp?$model/maxout10/MatMul/ReadVariableOp?!model/maxout10/add/ReadVariableOp?
#model/maxout0/MatMul/ReadVariableOpReadVariableOp,model_maxout0_matmul_readvariableop_resource*
_output_shapes
:	)?*
dtype02%
#model/maxout0/MatMul/ReadVariableOp?
model/maxout0/MatMulMatMulinput+model/maxout0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/maxout0/MatMul?
 model/maxout0/add/ReadVariableOpReadVariableOp)model_maxout0_add_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 model/maxout0/add/ReadVariableOp?
model/maxout0/addAddV2model/maxout0/MatMul:product:0(model/maxout0/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/maxout0/add?
model/maxout0/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   H   2
model/maxout0/Reshape/shape?
model/maxout0/ReshapeReshapemodel/maxout0/add:z:0$model/maxout0/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????H2
model/maxout0/Reshape?
model/maxout0/maxout/ShapeShapemodel/maxout0/Reshape:output:0*
T0*
_output_shapes
:2
model/maxout0/maxout/Shape?
(model/maxout0/maxout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model/maxout0/maxout/strided_slice/stack?
*model/maxout0/maxout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/maxout0/maxout/strided_slice/stack_1?
*model/maxout0/maxout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/maxout0/maxout/strided_slice/stack_2?
"model/maxout0/maxout/strided_sliceStridedSlice#model/maxout0/maxout/Shape:output:01model/maxout0/maxout/strided_slice/stack:output:03model/maxout0/maxout/strided_slice/stack_1:output:03model/maxout0/maxout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"model/maxout0/maxout/strided_slice?
$model/maxout0/maxout/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model/maxout0/maxout/Reshape/shape/1?
$model/maxout0/maxout/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model/maxout0/maxout/Reshape/shape/2?
$model/maxout0/maxout/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :H2&
$model/maxout0/maxout/Reshape/shape/3?
"model/maxout0/maxout/Reshape/shapePack+model/maxout0/maxout/strided_slice:output:0-model/maxout0/maxout/Reshape/shape/1:output:0-model/maxout0/maxout/Reshape/shape/2:output:0-model/maxout0/maxout/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"model/maxout0/maxout/Reshape/shape?
model/maxout0/maxout/ReshapeReshapemodel/maxout0/Reshape:output:0+model/maxout0/maxout/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????H2
model/maxout0/maxout/Reshape?
*model/maxout0/maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*model/maxout0/maxout/Max/reduction_indices?
model/maxout0/maxout/MaxMax%model/maxout0/maxout/Reshape:output:03model/maxout0/maxout/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????H2
model/maxout0/maxout/Max?
model/maxout0/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????H   2
model/maxout0/Reshape_1/shape?
model/maxout0/Reshape_1Reshape!model/maxout0/maxout/Max:output:0&model/maxout0/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????H2
model/maxout0/Reshape_1?
model/BN1/add/ReadVariableOpReadVariableOp%model_bn1_add_readvariableop_resource*
_output_shapes
:H*
dtype02
model/BN1/add/ReadVariableOp?
model/BN1/addAddV2 model/maxout0/Reshape_1:output:0$model/BN1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model/BN1/add?
model/BN1/MatMul/ReadVariableOpReadVariableOp(model_bn1_matmul_readvariableop_resource*
_output_shapes

:HH*
dtype02!
model/BN1/MatMul/ReadVariableOp?
model/BN1/MatMulMatMul model/maxout0/Reshape_1:output:0'model/BN1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model/BN1/MatMul?
"model/dense2/MatMul/ReadVariableOpReadVariableOp+model_dense2_matmul_readvariableop_resource*
_output_shapes

:H9*
dtype02$
"model/dense2/MatMul/ReadVariableOp?
model/dense2/MatMulMatMulmodel/BN1/MatMul:product:0*model/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
model/dense2/MatMul?
#model/dense2/BiasAdd/ReadVariableOpReadVariableOp,model_dense2_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype02%
#model/dense2/BiasAdd/ReadVariableOp?
model/dense2/BiasAddBiasAddmodel/dense2/MatMul:product:0+model/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
model/dense2/BiasAdd
model/dense2/ReluRelumodel/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????92
model/dense2/Relu?
model/BN3/add/ReadVariableOpReadVariableOp%model_bn3_add_readvariableop_resource*
_output_shapes
:9*
dtype02
model/BN3/add/ReadVariableOp?
model/BN3/addAddV2model/dense2/Relu:activations:0$model/BN3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
model/BN3/add?
model/BN3/MatMul/ReadVariableOpReadVariableOp(model_bn3_matmul_readvariableop_resource*
_output_shapes

:99*
dtype02!
model/BN3/MatMul/ReadVariableOp?
model/BN3/MatMulMatMulmodel/dense2/Relu:activations:0'model/BN3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
model/BN3/MatMul?
"model/dense4/MatMul/ReadVariableOpReadVariableOp+model_dense4_matmul_readvariableop_resource*
_output_shapes

:9<*
dtype02$
"model/dense4/MatMul/ReadVariableOp?
model/dense4/MatMulMatMulmodel/BN3/MatMul:product:0*model/dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model/dense4/MatMul?
#model/dense4/BiasAdd/ReadVariableOpReadVariableOp,model_dense4_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02%
#model/dense4/BiasAdd/ReadVariableOp?
model/dense4/BiasAddBiasAddmodel/dense4/MatMul:product:0+model/dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model/dense4/BiasAdd
model/dense4/ReluRelumodel/dense4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
model/dense4/Relu?
model/BN5/add/ReadVariableOpReadVariableOp%model_bn5_add_readvariableop_resource*
_output_shapes
:<*
dtype02
model/BN5/add/ReadVariableOp?
model/BN5/addAddV2model/dense4/Relu:activations:0$model/BN5/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model/BN5/add?
model/BN5/MatMul/ReadVariableOpReadVariableOp(model_bn5_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02!
model/BN5/MatMul/ReadVariableOp?
model/BN5/MatMulMatMulmodel/dense4/Relu:activations:0'model/BN5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
model/BN5/MatMul?
"model/dense6/MatMul/ReadVariableOpReadVariableOp+model_dense6_matmul_readvariableop_resource*
_output_shapes

:<0*
dtype02$
"model/dense6/MatMul/ReadVariableOp?
model/dense6/MatMulMatMulmodel/BN5/MatMul:product:0*model/dense6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense6/MatMul?
#model/dense6/BiasAdd/ReadVariableOpReadVariableOp,model_dense6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02%
#model/dense6/BiasAdd/ReadVariableOp?
model/dense6/BiasAddBiasAddmodel/dense6/MatMul:product:0+model/dense6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense6/BiasAdd
model/dense6/ReluRelumodel/dense6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
model/dense6/Relu?
model/BN7/add/ReadVariableOpReadVariableOp%model_bn7_add_readvariableop_resource*
_output_shapes
:0*
dtype02
model/BN7/add/ReadVariableOp?
model/BN7/addAddV2model/dense6/Relu:activations:0$model/BN7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/BN7/add?
model/BN7/MatMul/ReadVariableOpReadVariableOp(model_bn7_matmul_readvariableop_resource*
_output_shapes

:00*
dtype02!
model/BN7/MatMul/ReadVariableOp?
model/BN7/MatMulMatMulmodel/dense6/Relu:activations:0'model/BN7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/BN7/MatMul?
"model/dense8/MatMul/ReadVariableOpReadVariableOp+model_dense8_matmul_readvariableop_resource*
_output_shapes

:0$*
dtype02$
"model/dense8/MatMul/ReadVariableOp?
model/dense8/MatMulMatMulmodel/BN7/MatMul:product:0*model/dense8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
model/dense8/MatMul?
#model/dense8/BiasAdd/ReadVariableOpReadVariableOp,model_dense8_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#model/dense8/BiasAdd/ReadVariableOp?
model/dense8/BiasAddBiasAddmodel/dense8/MatMul:product:0+model/dense8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
model/dense8/BiasAdd
model/dense8/ReluRelumodel/dense8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
model/dense8/Relu?
model/BN9/add/ReadVariableOpReadVariableOp%model_bn9_add_readvariableop_resource*
_output_shapes
:$*
dtype02
model/BN9/add/ReadVariableOp?
model/BN9/addAddV2model/dense8/Relu:activations:0$model/BN9/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
model/BN9/add?
model/BN9/MatMul/ReadVariableOpReadVariableOp(model_bn9_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02!
model/BN9/MatMul/ReadVariableOp?
model/BN9/MatMulMatMulmodel/dense8/Relu:activations:0'model/BN9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
model/BN9/MatMul?
$model/maxout10/MatMul/ReadVariableOpReadVariableOp-model_maxout10_matmul_readvariableop_resource*
_output_shapes
:	$?*
dtype02&
$model/maxout10/MatMul/ReadVariableOp?
model/maxout10/MatMulMatMulmodel/BN9/MatMul:product:0,model/maxout10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/maxout10/MatMul?
!model/maxout10/add/ReadVariableOpReadVariableOp*model_maxout10_add_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/maxout10/add/ReadVariableOp?
model/maxout10/addAddV2model/maxout10/MatMul:product:0)model/maxout10/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/maxout10/add?
model/maxout10/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
model/maxout10/Reshape/shape?
model/maxout10/ReshapeReshapemodel/maxout10/add:z:0%model/maxout10/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
model/maxout10/Reshape?
model/maxout10/maxout_1/ShapeShapemodel/maxout10/Reshape:output:0*
T0*
_output_shapes
:2
model/maxout10/maxout_1/Shape?
+model/maxout10/maxout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model/maxout10/maxout_1/strided_slice/stack?
-model/maxout10/maxout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/maxout10/maxout_1/strided_slice/stack_1?
-model/maxout10/maxout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/maxout10/maxout_1/strided_slice/stack_2?
%model/maxout10/maxout_1/strided_sliceStridedSlice&model/maxout10/maxout_1/Shape:output:04model/maxout10/maxout_1/strided_slice/stack:output:06model/maxout10/maxout_1/strided_slice/stack_1:output:06model/maxout10/maxout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/maxout10/maxout_1/strided_slice?
'model/maxout10/maxout_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'model/maxout10/maxout_1/Reshape/shape/1?
'model/maxout10/maxout_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'model/maxout10/maxout_1/Reshape/shape/2?
'model/maxout10/maxout_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'model/maxout10/maxout_1/Reshape/shape/3?
%model/maxout10/maxout_1/Reshape/shapePack.model/maxout10/maxout_1/strided_slice:output:00model/maxout10/maxout_1/Reshape/shape/1:output:00model/maxout10/maxout_1/Reshape/shape/2:output:00model/maxout10/maxout_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%model/maxout10/maxout_1/Reshape/shape?
model/maxout10/maxout_1/ReshapeReshapemodel/maxout10/Reshape:output:0.model/maxout10/maxout_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
model/maxout10/maxout_1/Reshape?
-model/maxout10/maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-model/maxout10/maxout_1/Max/reduction_indices?
model/maxout10/maxout_1/MaxMax(model/maxout10/maxout_1/Reshape:output:06model/maxout10/maxout_1/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2
model/maxout10/maxout_1/Max?
model/maxout10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2 
model/maxout10/Reshape_1/shape?
model/maxout10/Reshape_1Reshape$model/maxout10/maxout_1/Max:output:0'model/maxout10/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
model/maxout10/Reshape_1?
model/BN11/add/ReadVariableOpReadVariableOp&model_bn11_add_readvariableop_resource*
_output_shapes
:*
dtype02
model/BN11/add/ReadVariableOp?
model/BN11/addAddV2!model/maxout10/Reshape_1:output:0%model/BN11/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/BN11/add?
 model/BN11/MatMul/ReadVariableOpReadVariableOp)model_bn11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 model/BN11/MatMul/ReadVariableOp?
model/BN11/MatMulMatMul!model/maxout10/Reshape_1:output:0(model/BN11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/BN11/MatMul?
#model/dense12/MatMul/ReadVariableOpReadVariableOp,model_dense12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense12/MatMul/ReadVariableOp?
model/dense12/MatMulMatMulmodel/BN11/MatMul:product:0+model/dense12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense12/MatMul?
$model/dense12/BiasAdd/ReadVariableOpReadVariableOp-model_dense12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense12/BiasAdd/ReadVariableOp?
model/dense12/BiasAddBiasAddmodel/dense12/MatMul:product:0,model/dense12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense12/BiasAdd?
model/dense12/ReluRelumodel/dense12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense12/Relu?
model/BN13/add/ReadVariableOpReadVariableOp&model_bn13_add_readvariableop_resource*
_output_shapes
:*
dtype02
model/BN13/add/ReadVariableOp?
model/BN13/addAddV2 model/dense12/Relu:activations:0%model/BN13/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/BN13/add?
 model/BN13/MatMul/ReadVariableOpReadVariableOp)model_bn13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 model/BN13/MatMul/ReadVariableOp?
model/BN13/MatMulMatMul model/dense12/Relu:activations:0(model/BN13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/BN13/MatMul?
#model/dense14/MatMul/ReadVariableOpReadVariableOp,model_dense14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense14/MatMul/ReadVariableOp?
model/dense14/MatMulMatMulmodel/BN13/MatMul:product:0+model/dense14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense14/MatMul?
$model/dense14/BiasAdd/ReadVariableOpReadVariableOp-model_dense14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense14/BiasAdd/ReadVariableOp?
model/dense14/BiasAddBiasAddmodel/dense14/MatMul:product:0,model/dense14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense14/BiasAdd?
model/dense14/ReluRelumodel/dense14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense14/Relu?
model/BN15/add/ReadVariableOpReadVariableOp&model_bn15_add_readvariableop_resource*
_output_shapes
:*
dtype02
model/BN15/add/ReadVariableOp?
model/BN15/addAddV2 model/dense14/Relu:activations:0%model/BN15/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/BN15/add?
 model/BN15/MatMul/ReadVariableOpReadVariableOp)model_bn15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 model/BN15/MatMul/ReadVariableOp?
model/BN15/MatMulMatMul model/dense14/Relu:activations:0(model/BN15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/BN15/MatMul?
#model/dense16/MatMul/ReadVariableOpReadVariableOp,model_dense16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense16/MatMul/ReadVariableOp?
model/dense16/MatMulMatMulmodel/BN15/MatMul:product:0+model/dense16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense16/MatMul?
$model/dense16/BiasAdd/ReadVariableOpReadVariableOp-model_dense16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense16/BiasAdd/ReadVariableOp?
model/dense16/BiasAddBiasAddmodel/dense16/MatMul:product:0,model/dense16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense16/BiasAdd?
model/dense16/SoftmaxSoftmaxmodel/dense16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense16/Softmax?

IdentityIdentitymodel/dense16/Softmax:softmax:0 ^model/BN1/MatMul/ReadVariableOp^model/BN1/add/ReadVariableOp!^model/BN11/MatMul/ReadVariableOp^model/BN11/add/ReadVariableOp!^model/BN13/MatMul/ReadVariableOp^model/BN13/add/ReadVariableOp!^model/BN15/MatMul/ReadVariableOp^model/BN15/add/ReadVariableOp ^model/BN3/MatMul/ReadVariableOp^model/BN3/add/ReadVariableOp ^model/BN5/MatMul/ReadVariableOp^model/BN5/add/ReadVariableOp ^model/BN7/MatMul/ReadVariableOp^model/BN7/add/ReadVariableOp ^model/BN9/MatMul/ReadVariableOp^model/BN9/add/ReadVariableOp%^model/dense12/BiasAdd/ReadVariableOp$^model/dense12/MatMul/ReadVariableOp%^model/dense14/BiasAdd/ReadVariableOp$^model/dense14/MatMul/ReadVariableOp%^model/dense16/BiasAdd/ReadVariableOp$^model/dense16/MatMul/ReadVariableOp$^model/dense2/BiasAdd/ReadVariableOp#^model/dense2/MatMul/ReadVariableOp$^model/dense4/BiasAdd/ReadVariableOp#^model/dense4/MatMul/ReadVariableOp$^model/dense6/BiasAdd/ReadVariableOp#^model/dense6/MatMul/ReadVariableOp$^model/dense8/BiasAdd/ReadVariableOp#^model/dense8/MatMul/ReadVariableOp$^model/maxout0/MatMul/ReadVariableOp!^model/maxout0/add/ReadVariableOp%^model/maxout10/MatMul/ReadVariableOp"^model/maxout10/add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::2B
model/BN1/MatMul/ReadVariableOpmodel/BN1/MatMul/ReadVariableOp2<
model/BN1/add/ReadVariableOpmodel/BN1/add/ReadVariableOp2D
 model/BN11/MatMul/ReadVariableOp model/BN11/MatMul/ReadVariableOp2>
model/BN11/add/ReadVariableOpmodel/BN11/add/ReadVariableOp2D
 model/BN13/MatMul/ReadVariableOp model/BN13/MatMul/ReadVariableOp2>
model/BN13/add/ReadVariableOpmodel/BN13/add/ReadVariableOp2D
 model/BN15/MatMul/ReadVariableOp model/BN15/MatMul/ReadVariableOp2>
model/BN15/add/ReadVariableOpmodel/BN15/add/ReadVariableOp2B
model/BN3/MatMul/ReadVariableOpmodel/BN3/MatMul/ReadVariableOp2<
model/BN3/add/ReadVariableOpmodel/BN3/add/ReadVariableOp2B
model/BN5/MatMul/ReadVariableOpmodel/BN5/MatMul/ReadVariableOp2<
model/BN5/add/ReadVariableOpmodel/BN5/add/ReadVariableOp2B
model/BN7/MatMul/ReadVariableOpmodel/BN7/MatMul/ReadVariableOp2<
model/BN7/add/ReadVariableOpmodel/BN7/add/ReadVariableOp2B
model/BN9/MatMul/ReadVariableOpmodel/BN9/MatMul/ReadVariableOp2<
model/BN9/add/ReadVariableOpmodel/BN9/add/ReadVariableOp2L
$model/dense12/BiasAdd/ReadVariableOp$model/dense12/BiasAdd/ReadVariableOp2J
#model/dense12/MatMul/ReadVariableOp#model/dense12/MatMul/ReadVariableOp2L
$model/dense14/BiasAdd/ReadVariableOp$model/dense14/BiasAdd/ReadVariableOp2J
#model/dense14/MatMul/ReadVariableOp#model/dense14/MatMul/ReadVariableOp2L
$model/dense16/BiasAdd/ReadVariableOp$model/dense16/BiasAdd/ReadVariableOp2J
#model/dense16/MatMul/ReadVariableOp#model/dense16/MatMul/ReadVariableOp2J
#model/dense2/BiasAdd/ReadVariableOp#model/dense2/BiasAdd/ReadVariableOp2H
"model/dense2/MatMul/ReadVariableOp"model/dense2/MatMul/ReadVariableOp2J
#model/dense4/BiasAdd/ReadVariableOp#model/dense4/BiasAdd/ReadVariableOp2H
"model/dense4/MatMul/ReadVariableOp"model/dense4/MatMul/ReadVariableOp2J
#model/dense6/BiasAdd/ReadVariableOp#model/dense6/BiasAdd/ReadVariableOp2H
"model/dense6/MatMul/ReadVariableOp"model/dense6/MatMul/ReadVariableOp2J
#model/dense8/BiasAdd/ReadVariableOp#model/dense8/BiasAdd/ReadVariableOp2H
"model/dense8/MatMul/ReadVariableOp"model/dense8/MatMul/ReadVariableOp2J
#model/maxout0/MatMul/ReadVariableOp#model/maxout0/MatMul/ReadVariableOp2D
 model/maxout0/add/ReadVariableOp model/maxout0/add/ReadVariableOp2L
$model/maxout10/MatMul/ReadVariableOp$model/maxout10/MatMul/ReadVariableOp2F
!model/maxout10/add/ReadVariableOp!model/maxout10/add/ReadVariableOp:N J
'
_output_shapes
:?????????)

_user_specified_nameinput
?
?
A__inference_maxout0_layer_call_and_return_conditional_losses_2118

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	)?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpt
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
adds
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   H   2
Reshape/shapet
ReshapeReshapeadd:z:0Reshape/shape:output:0*
T0*+
_output_shapes
:?????????H2	
Reshape\
maxout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:2
maxout/Shape?
maxout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
maxout/strided_slice/stack?
maxout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
maxout/strided_slice/stack_1?
maxout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
maxout/strided_slice/stack_2?
maxout/strided_sliceStridedSlicemaxout/Shape:output:0#maxout/strided_slice/stack:output:0%maxout/strided_slice/stack_1:output:0%maxout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
maxout/strided_slicer
maxout/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/1r
maxout/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/2r
maxout/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :H2
maxout/Reshape/shape/3?
maxout/Reshape/shapePackmaxout/strided_slice:output:0maxout/Reshape/shape/1:output:0maxout/Reshape/shape/2:output:0maxout/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
maxout/Reshape/shape?
maxout/ReshapeReshapeReshape:output:0maxout/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????H2
maxout/Reshape~
maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
maxout/Max/reduction_indices?

maxout/MaxMaxmaxout/Reshape:output:0%maxout/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????H2

maxout/Maxs
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????H   2
Reshape_1/shape?
	Reshape_1Reshapemaxout/Max:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????H2
	Reshape_1?
IdentityIdentityReshape_1:output:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????)::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
?
=__inference_BN3_layer_call_and_return_conditional_losses_2176

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:9*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:99*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????92

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
>__inference_BN15_layer_call_and_return_conditional_losses_1123

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?P
?	
?__inference_model_layer_call_and_return_conditional_losses_1167	
input
maxout0_722
maxout0_724
bn1_748
bn1_750

dense2_775

dense2_777
bn3_801
bn3_803

dense4_828

dense4_830
bn5_854
bn5_856

dense6_881

dense6_883
bn7_907
bn7_909

dense8_934

dense8_936
bn9_960
bn9_962
maxout10_1002
maxout10_1004
	bn11_1028
	bn11_1030
dense12_1055
dense12_1057
	bn13_1081
	bn13_1083
dense14_1108
dense14_1110
	bn15_1134
	bn15_1136
dense16_1161
dense16_1163
identity??BN1/StatefulPartitionedCall?BN11/StatefulPartitionedCall?BN13/StatefulPartitionedCall?BN15/StatefulPartitionedCall?BN3/StatefulPartitionedCall?BN5/StatefulPartitionedCall?BN7/StatefulPartitionedCall?BN9/StatefulPartitionedCall?dense12/StatefulPartitionedCall?dense14/StatefulPartitionedCall?dense16/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense4/StatefulPartitionedCall?dense6/StatefulPartitionedCall?dense8/StatefulPartitionedCall?maxout0/StatefulPartitionedCall? maxout10/StatefulPartitionedCall?
maxout0/StatefulPartitionedCallStatefulPartitionedCallinputmaxout0_722maxout0_724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_maxout0_layer_call_and_return_conditional_losses_7112!
maxout0/StatefulPartitionedCall?
BN1/StatefulPartitionedCallStatefulPartitionedCall(maxout0/StatefulPartitionedCall:output:0bn1_748bn1_750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN1_layer_call_and_return_conditional_losses_7372
BN1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall$BN1/StatefulPartitionedCall:output:0
dense2_775
dense2_777*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense2_layer_call_and_return_conditional_losses_7642 
dense2/StatefulPartitionedCall?
BN3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0bn3_801bn3_803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN3_layer_call_and_return_conditional_losses_7902
BN3/StatefulPartitionedCall?
dense4/StatefulPartitionedCallStatefulPartitionedCall$BN3/StatefulPartitionedCall:output:0
dense4_828
dense4_830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense4_layer_call_and_return_conditional_losses_8172 
dense4/StatefulPartitionedCall?
BN5/StatefulPartitionedCallStatefulPartitionedCall'dense4/StatefulPartitionedCall:output:0bn5_854bn5_856*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN5_layer_call_and_return_conditional_losses_8432
BN5/StatefulPartitionedCall?
dense6/StatefulPartitionedCallStatefulPartitionedCall$BN5/StatefulPartitionedCall:output:0
dense6_881
dense6_883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense6_layer_call_and_return_conditional_losses_8702 
dense6/StatefulPartitionedCall?
BN7/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0bn7_907bn7_909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN7_layer_call_and_return_conditional_losses_8962
BN7/StatefulPartitionedCall?
dense8/StatefulPartitionedCallStatefulPartitionedCall$BN7/StatefulPartitionedCall:output:0
dense8_934
dense8_936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense8_layer_call_and_return_conditional_losses_9232 
dense8/StatefulPartitionedCall?
BN9/StatefulPartitionedCallStatefulPartitionedCall'dense8/StatefulPartitionedCall:output:0bn9_960bn9_962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN9_layer_call_and_return_conditional_losses_9492
BN9/StatefulPartitionedCall?
 maxout10/StatefulPartitionedCallStatefulPartitionedCall$BN9/StatefulPartitionedCall:output:0maxout10_1002maxout10_1004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_maxout10_layer_call_and_return_conditional_losses_9912"
 maxout10/StatefulPartitionedCall?
BN11/StatefulPartitionedCallStatefulPartitionedCall)maxout10/StatefulPartitionedCall:output:0	bn11_1028	bn11_1030*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN11_layer_call_and_return_conditional_losses_10172
BN11/StatefulPartitionedCall?
dense12/StatefulPartitionedCallStatefulPartitionedCall%BN11/StatefulPartitionedCall:output:0dense12_1055dense12_1057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense12_layer_call_and_return_conditional_losses_10442!
dense12/StatefulPartitionedCall?
BN13/StatefulPartitionedCallStatefulPartitionedCall(dense12/StatefulPartitionedCall:output:0	bn13_1081	bn13_1083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN13_layer_call_and_return_conditional_losses_10702
BN13/StatefulPartitionedCall?
dense14/StatefulPartitionedCallStatefulPartitionedCall%BN13/StatefulPartitionedCall:output:0dense14_1108dense14_1110*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense14_layer_call_and_return_conditional_losses_10972!
dense14/StatefulPartitionedCall?
BN15/StatefulPartitionedCallStatefulPartitionedCall(dense14/StatefulPartitionedCall:output:0	bn15_1134	bn15_1136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN15_layer_call_and_return_conditional_losses_11232
BN15/StatefulPartitionedCall?
dense16/StatefulPartitionedCallStatefulPartitionedCall%BN15/StatefulPartitionedCall:output:0dense16_1161dense16_1163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense16_layer_call_and_return_conditional_losses_11502!
dense16/StatefulPartitionedCall?
IdentityIdentity(dense16/StatefulPartitionedCall:output:0^BN1/StatefulPartitionedCall^BN11/StatefulPartitionedCall^BN13/StatefulPartitionedCall^BN15/StatefulPartitionedCall^BN3/StatefulPartitionedCall^BN5/StatefulPartitionedCall^BN7/StatefulPartitionedCall^BN9/StatefulPartitionedCall ^dense12/StatefulPartitionedCall ^dense14/StatefulPartitionedCall ^dense16/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense4/StatefulPartitionedCall^dense6/StatefulPartitionedCall^dense8/StatefulPartitionedCall ^maxout0/StatefulPartitionedCall!^maxout10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::2:
BN1/StatefulPartitionedCallBN1/StatefulPartitionedCall2<
BN11/StatefulPartitionedCallBN11/StatefulPartitionedCall2<
BN13/StatefulPartitionedCallBN13/StatefulPartitionedCall2<
BN15/StatefulPartitionedCallBN15/StatefulPartitionedCall2:
BN3/StatefulPartitionedCallBN3/StatefulPartitionedCall2:
BN5/StatefulPartitionedCallBN5/StatefulPartitionedCall2:
BN7/StatefulPartitionedCallBN7/StatefulPartitionedCall2:
BN9/StatefulPartitionedCallBN9/StatefulPartitionedCall2B
dense12/StatefulPartitionedCalldense12/StatefulPartitionedCall2B
dense14/StatefulPartitionedCalldense14/StatefulPartitionedCall2B
dense16/StatefulPartitionedCalldense16/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall2@
dense8/StatefulPartitionedCalldense8/StatefulPartitionedCall2B
maxout0/StatefulPartitionedCallmaxout0/StatefulPartitionedCall2D
 maxout10/StatefulPartitionedCall maxout10/StatefulPartitionedCall:N J
'
_output_shapes
:?????????)

_user_specified_nameinput
?
?
"__inference_signature_wrapper_1656	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__wrapped_model_6812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????)

_user_specified_nameinput
?
?
=__inference_BN9_layer_call_and_return_conditional_losses_2293

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:$*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs
?
x
#__inference_BN11_layer_call_fn_2356

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN11_layer_call_and_return_conditional_losses_10172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_dense16_layer_call_fn_2454

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense16_layer_call_and_return_conditional_losses_11502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
?__inference_dense4_layer_call_and_return_conditional_losses_817

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:9<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
<__inference_BN5_layer_call_and_return_conditional_losses_843

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:<*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
A__inference_maxout10_layer_call_and_return_conditional_losses_991

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	$?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpt
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
adds
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shapet
ReshapeReshapeadd:z:0Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshape\
maxout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:2
maxout/Shape?
maxout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
maxout/strided_slice/stack?
maxout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
maxout/strided_slice/stack_1?
maxout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
maxout/strided_slice/stack_2?
maxout/strided_sliceStridedSlicemaxout/Shape:output:0#maxout/strided_slice/stack:output:0%maxout/strided_slice/stack_1:output:0%maxout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
maxout/strided_slicer
maxout/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/1r
maxout/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/2r
maxout/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/3?
maxout/Reshape/shapePackmaxout/strided_slice:output:0maxout/Reshape/shape/1:output:0maxout/Reshape/shape/2:output:0maxout/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
maxout/Reshape/shape?
maxout/ReshapeReshapeReshape:output:0maxout/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
maxout/Reshape~
maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
maxout/Max/reduction_indices?

maxout/MaxMaxmaxout/Reshape:output:0%maxout/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2

maxout/Maxs
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape?
	Reshape_1Reshapemaxout/Max:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1?
IdentityIdentityReshape_1:output:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs
?
?
=__inference_BN1_layer_call_and_return_conditional_losses_2137

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:H*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:HH*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????H::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?	
?
A__inference_dense14_layer_call_and_return_conditional_losses_2406

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_dense12_layer_call_and_return_conditional_losses_2367

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
?__inference_dense2_layer_call_and_return_conditional_losses_764

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H9*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:9*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????92
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????92

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????H::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
z
%__inference_dense8_layer_call_fn_2283

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense8_layer_call_and_return_conditional_losses_9232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
׊
?
 __inference__traced_restore_2691
file_prefix%
!assignvariableop_maxout0_maxout_w'
#assignvariableop_1_maxout0_maxout_b!
assignvariableop_2_bn1_norm_w!
assignvariableop_3_bn1_norm_b$
 assignvariableop_4_dense2_kernel"
assignvariableop_5_dense2_bias!
assignvariableop_6_bn3_norm_w!
assignvariableop_7_bn3_norm_b$
 assignvariableop_8_dense4_kernel"
assignvariableop_9_dense4_bias"
assignvariableop_10_bn5_norm_w"
assignvariableop_11_bn5_norm_b%
!assignvariableop_12_dense6_kernel#
assignvariableop_13_dense6_bias"
assignvariableop_14_bn7_norm_w"
assignvariableop_15_bn7_norm_b%
!assignvariableop_16_dense8_kernel#
assignvariableop_17_dense8_bias"
assignvariableop_18_bn9_norm_w"
assignvariableop_19_bn9_norm_b)
%assignvariableop_20_maxout10_maxout_w)
%assignvariableop_21_maxout10_maxout_b#
assignvariableop_22_bn11_norm_w#
assignvariableop_23_bn11_norm_b&
"assignvariableop_24_dense12_kernel$
 assignvariableop_25_dense12_bias#
assignvariableop_26_bn13_norm_w#
assignvariableop_27_bn13_norm_b&
"assignvariableop_28_dense14_kernel$
 assignvariableop_29_dense14_bias#
assignvariableop_30_bn15_norm_w#
assignvariableop_31_bn15_norm_b&
"assignvariableop_32_dense16_kernel$
 assignvariableop_33_dense16_bias
identity_35??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B8layer_with_weights-0/maxout_w/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/maxout_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/norm_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/norm_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/norm_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/norm_b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/norm_w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/norm_b/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-10/maxout_w/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-10/maxout_b/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/norm_w/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/norm_b/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/norm_w/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/norm_b/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/norm_w/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/norm_b/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_maxout0_maxout_wIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_maxout0_maxout_bIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn1_norm_wIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn1_norm_bIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_bn3_norm_wIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_bn3_norm_bIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_bn5_norm_wIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_bn5_norm_bIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_bn7_norm_wIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_bn7_norm_bIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_dense8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_bn9_norm_wIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_bn9_norm_bIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_maxout10_maxout_wIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_maxout10_maxout_bIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_bn11_norm_wIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_bn11_norm_bIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense12_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense12_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_bn13_norm_wIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_bn13_norm_bIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense14_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp assignvariableop_29_dense14_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_bn15_norm_wIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_bn15_norm_bIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense16_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense16_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34?
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_35"#
identity_35Identity_35:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
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
?
w
"__inference_BN1_layer_call_fn_2146

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_BN1_layer_call_and_return_conditional_losses_7372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????H::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
B__inference_maxout10_layer_call_and_return_conditional_losses_2328

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	$?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpt
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
adds
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shapet
ReshapeReshapeadd:z:0Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshape\
maxout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:2
maxout/Shape?
maxout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
maxout/strided_slice/stack?
maxout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
maxout/strided_slice/stack_1?
maxout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
maxout/strided_slice/stack_2?
maxout/strided_sliceStridedSlicemaxout/Shape:output:0#maxout/strided_slice/stack:output:0%maxout/strided_slice/stack_1:output:0%maxout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
maxout/strided_slicer
maxout/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/1r
maxout/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/2r
maxout/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
maxout/Reshape/shape/3?
maxout/Reshape/shapePackmaxout/strided_slice:output:0maxout/Reshape/shape/1:output:0maxout/Reshape/shape/2:output:0maxout/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
maxout/Reshape/shape?
maxout/ReshapeReshapeReshape:output:0maxout/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
maxout/Reshape~
maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
maxout/Max/reduction_indices?

maxout/MaxMaxmaxout/Reshape:output:0%maxout/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2

maxout/Maxs
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape?
	Reshape_1Reshapemaxout/Max:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1?
IdentityIdentityReshape_1:output:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs
?	
?
?__inference_dense8_layer_call_and_return_conditional_losses_923

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
x
#__inference_BN15_layer_call_fn_2434

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_BN15_layer_call_and_return_conditional_losses_11232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
<__inference_BN1_layer_call_and_return_conditional_losses_737

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:H*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:HH*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????H::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_1419	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_13482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????)

_user_specified_nameinput
?	
?
@__inference_dense8_layer_call_and_return_conditional_losses_2274

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?	
?
@__inference_dense2_layer_call_and_return_conditional_losses_2157

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H9*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:9*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????92
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????92

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????H::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
>__inference_BN11_layer_call_and_return_conditional_losses_1017

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
?__inference_dense6_layer_call_and_return_conditional_losses_870

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?	
?
A__inference_dense12_layer_call_and_return_conditional_losses_1044

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_dense14_layer_call_and_return_conditional_losses_1097

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_2092

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_15102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_1801

inputs*
&maxout0_matmul_readvariableop_resource'
#maxout0_add_readvariableop_resource#
bn1_add_readvariableop_resource&
"bn1_matmul_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource#
bn3_add_readvariableop_resource&
"bn3_matmul_readvariableop_resource)
%dense4_matmul_readvariableop_resource*
&dense4_biasadd_readvariableop_resource#
bn5_add_readvariableop_resource&
"bn5_matmul_readvariableop_resource)
%dense6_matmul_readvariableop_resource*
&dense6_biasadd_readvariableop_resource#
bn7_add_readvariableop_resource&
"bn7_matmul_readvariableop_resource)
%dense8_matmul_readvariableop_resource*
&dense8_biasadd_readvariableop_resource#
bn9_add_readvariableop_resource&
"bn9_matmul_readvariableop_resource+
'maxout10_matmul_readvariableop_resource(
$maxout10_add_readvariableop_resource$
 bn11_add_readvariableop_resource'
#bn11_matmul_readvariableop_resource*
&dense12_matmul_readvariableop_resource+
'dense12_biasadd_readvariableop_resource$
 bn13_add_readvariableop_resource'
#bn13_matmul_readvariableop_resource*
&dense14_matmul_readvariableop_resource+
'dense14_biasadd_readvariableop_resource$
 bn15_add_readvariableop_resource'
#bn15_matmul_readvariableop_resource*
&dense16_matmul_readvariableop_resource+
'dense16_biasadd_readvariableop_resource
identity??BN1/MatMul/ReadVariableOp?BN1/add/ReadVariableOp?BN11/MatMul/ReadVariableOp?BN11/add/ReadVariableOp?BN13/MatMul/ReadVariableOp?BN13/add/ReadVariableOp?BN15/MatMul/ReadVariableOp?BN15/add/ReadVariableOp?BN3/MatMul/ReadVariableOp?BN3/add/ReadVariableOp?BN5/MatMul/ReadVariableOp?BN5/add/ReadVariableOp?BN7/MatMul/ReadVariableOp?BN7/add/ReadVariableOp?BN9/MatMul/ReadVariableOp?BN9/add/ReadVariableOp?dense12/BiasAdd/ReadVariableOp?dense12/MatMul/ReadVariableOp?dense14/BiasAdd/ReadVariableOp?dense14/MatMul/ReadVariableOp?dense16/BiasAdd/ReadVariableOp?dense16/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?dense4/BiasAdd/ReadVariableOp?dense4/MatMul/ReadVariableOp?dense6/BiasAdd/ReadVariableOp?dense6/MatMul/ReadVariableOp?dense8/BiasAdd/ReadVariableOp?dense8/MatMul/ReadVariableOp?maxout0/MatMul/ReadVariableOp?maxout0/add/ReadVariableOp?maxout10/MatMul/ReadVariableOp?maxout10/add/ReadVariableOp?
maxout0/MatMul/ReadVariableOpReadVariableOp&maxout0_matmul_readvariableop_resource*
_output_shapes
:	)?*
dtype02
maxout0/MatMul/ReadVariableOp?
maxout0/MatMulMatMulinputs%maxout0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
maxout0/MatMul?
maxout0/add/ReadVariableOpReadVariableOp#maxout0_add_readvariableop_resource*
_output_shapes	
:?*
dtype02
maxout0/add/ReadVariableOp?
maxout0/addAddV2maxout0/MatMul:product:0"maxout0/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
maxout0/add?
maxout0/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   H   2
maxout0/Reshape/shape?
maxout0/ReshapeReshapemaxout0/add:z:0maxout0/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????H2
maxout0/Reshapet
maxout0/maxout/ShapeShapemaxout0/Reshape:output:0*
T0*
_output_shapes
:2
maxout0/maxout/Shape?
"maxout0/maxout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"maxout0/maxout/strided_slice/stack?
$maxout0/maxout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$maxout0/maxout/strided_slice/stack_1?
$maxout0/maxout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$maxout0/maxout/strided_slice/stack_2?
maxout0/maxout/strided_sliceStridedSlicemaxout0/maxout/Shape:output:0+maxout0/maxout/strided_slice/stack:output:0-maxout0/maxout/strided_slice/stack_1:output:0-maxout0/maxout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
maxout0/maxout/strided_slice?
maxout0/maxout/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
maxout0/maxout/Reshape/shape/1?
maxout0/maxout/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
maxout0/maxout/Reshape/shape/2?
maxout0/maxout/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :H2 
maxout0/maxout/Reshape/shape/3?
maxout0/maxout/Reshape/shapePack%maxout0/maxout/strided_slice:output:0'maxout0/maxout/Reshape/shape/1:output:0'maxout0/maxout/Reshape/shape/2:output:0'maxout0/maxout/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
maxout0/maxout/Reshape/shape?
maxout0/maxout/ReshapeReshapemaxout0/Reshape:output:0%maxout0/maxout/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????H2
maxout0/maxout/Reshape?
$maxout0/maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2&
$maxout0/maxout/Max/reduction_indices?
maxout0/maxout/MaxMaxmaxout0/maxout/Reshape:output:0-maxout0/maxout/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????H2
maxout0/maxout/Max?
maxout0/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????H   2
maxout0/Reshape_1/shape?
maxout0/Reshape_1Reshapemaxout0/maxout/Max:output:0 maxout0/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????H2
maxout0/Reshape_1?
BN1/add/ReadVariableOpReadVariableOpbn1_add_readvariableop_resource*
_output_shapes
:H*
dtype02
BN1/add/ReadVariableOp?
BN1/addAddV2maxout0/Reshape_1:output:0BN1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BN1/add?
BN1/MatMul/ReadVariableOpReadVariableOp"bn1_matmul_readvariableop_resource*
_output_shapes

:HH*
dtype02
BN1/MatMul/ReadVariableOp?

BN1/MatMulMatMulmaxout0/Reshape_1:output:0!BN1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2

BN1/MatMul?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:H9*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMulBN1/MatMul:product:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????92
dense2/Relu?
BN3/add/ReadVariableOpReadVariableOpbn3_add_readvariableop_resource*
_output_shapes
:9*
dtype02
BN3/add/ReadVariableOp?
BN3/addAddV2dense2/Relu:activations:0BN3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92	
BN3/add?
BN3/MatMul/ReadVariableOpReadVariableOp"bn3_matmul_readvariableop_resource*
_output_shapes

:99*
dtype02
BN3/MatMul/ReadVariableOp?

BN3/MatMulMatMuldense2/Relu:activations:0!BN3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????92

BN3/MatMul?
dense4/MatMul/ReadVariableOpReadVariableOp%dense4_matmul_readvariableop_resource*
_output_shapes

:9<*
dtype02
dense4/MatMul/ReadVariableOp?
dense4/MatMulMatMulBN3/MatMul:product:0$dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dense4/MatMul?
dense4/BiasAdd/ReadVariableOpReadVariableOp&dense4_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
dense4/BiasAdd/ReadVariableOp?
dense4/BiasAddBiasAdddense4/MatMul:product:0%dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dense4/BiasAddm
dense4/ReluReludense4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dense4/Relu?
BN5/add/ReadVariableOpReadVariableOpbn5_add_readvariableop_resource*
_output_shapes
:<*
dtype02
BN5/add/ReadVariableOp?
BN5/addAddV2dense4/Relu:activations:0BN5/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BN5/add?
BN5/MatMul/ReadVariableOpReadVariableOp"bn5_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
BN5/MatMul/ReadVariableOp?

BN5/MatMulMatMuldense4/Relu:activations:0!BN5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2

BN5/MatMul?
dense6/MatMul/ReadVariableOpReadVariableOp%dense6_matmul_readvariableop_resource*
_output_shapes

:<0*
dtype02
dense6/MatMul/ReadVariableOp?
dense6/MatMulMatMulBN5/MatMul:product:0$dense6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense6/MatMul?
dense6/BiasAdd/ReadVariableOpReadVariableOp&dense6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
dense6/BiasAdd/ReadVariableOp?
dense6/BiasAddBiasAdddense6/MatMul:product:0%dense6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense6/BiasAddm
dense6/ReluReludense6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
dense6/Relu?
BN7/add/ReadVariableOpReadVariableOpbn7_add_readvariableop_resource*
_output_shapes
:0*
dtype02
BN7/add/ReadVariableOp?
BN7/addAddV2dense6/Relu:activations:0BN7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BN7/add?
BN7/MatMul/ReadVariableOpReadVariableOp"bn7_matmul_readvariableop_resource*
_output_shapes

:00*
dtype02
BN7/MatMul/ReadVariableOp?

BN7/MatMulMatMuldense6/Relu:activations:0!BN7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02

BN7/MatMul?
dense8/MatMul/ReadVariableOpReadVariableOp%dense8_matmul_readvariableop_resource*
_output_shapes

:0$*
dtype02
dense8/MatMul/ReadVariableOp?
dense8/MatMulMatMulBN7/MatMul:product:0$dense8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense8/MatMul?
dense8/BiasAdd/ReadVariableOpReadVariableOp&dense8_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
dense8/BiasAdd/ReadVariableOp?
dense8/BiasAddBiasAdddense8/MatMul:product:0%dense8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense8/BiasAddm
dense8/ReluReludense8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
dense8/Relu?
BN9/add/ReadVariableOpReadVariableOpbn9_add_readvariableop_resource*
_output_shapes
:$*
dtype02
BN9/add/ReadVariableOp?
BN9/addAddV2dense8/Relu:activations:0BN9/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2	
BN9/add?
BN9/MatMul/ReadVariableOpReadVariableOp"bn9_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
BN9/MatMul/ReadVariableOp?

BN9/MatMulMatMuldense8/Relu:activations:0!BN9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2

BN9/MatMul?
maxout10/MatMul/ReadVariableOpReadVariableOp'maxout10_matmul_readvariableop_resource*
_output_shapes
:	$?*
dtype02 
maxout10/MatMul/ReadVariableOp?
maxout10/MatMulMatMulBN9/MatMul:product:0&maxout10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
maxout10/MatMul?
maxout10/add/ReadVariableOpReadVariableOp$maxout10_add_readvariableop_resource*
_output_shapes	
:?*
dtype02
maxout10/add/ReadVariableOp?
maxout10/addAddV2maxout10/MatMul:product:0#maxout10/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
maxout10/add?
maxout10/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
maxout10/Reshape/shape?
maxout10/ReshapeReshapemaxout10/add:z:0maxout10/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
maxout10/Reshape{
maxout10/maxout_1/ShapeShapemaxout10/Reshape:output:0*
T0*
_output_shapes
:2
maxout10/maxout_1/Shape?
%maxout10/maxout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%maxout10/maxout_1/strided_slice/stack?
'maxout10/maxout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'maxout10/maxout_1/strided_slice/stack_1?
'maxout10/maxout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'maxout10/maxout_1/strided_slice/stack_2?
maxout10/maxout_1/strided_sliceStridedSlice maxout10/maxout_1/Shape:output:0.maxout10/maxout_1/strided_slice/stack:output:00maxout10/maxout_1/strided_slice/stack_1:output:00maxout10/maxout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
maxout10/maxout_1/strided_slice?
!maxout10/maxout_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!maxout10/maxout_1/Reshape/shape/1?
!maxout10/maxout_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!maxout10/maxout_1/Reshape/shape/2?
!maxout10/maxout_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!maxout10/maxout_1/Reshape/shape/3?
maxout10/maxout_1/Reshape/shapePack(maxout10/maxout_1/strided_slice:output:0*maxout10/maxout_1/Reshape/shape/1:output:0*maxout10/maxout_1/Reshape/shape/2:output:0*maxout10/maxout_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
maxout10/maxout_1/Reshape/shape?
maxout10/maxout_1/ReshapeReshapemaxout10/Reshape:output:0(maxout10/maxout_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
maxout10/maxout_1/Reshape?
'maxout10/maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2)
'maxout10/maxout_1/Max/reduction_indices?
maxout10/maxout_1/MaxMax"maxout10/maxout_1/Reshape:output:00maxout10/maxout_1/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2
maxout10/maxout_1/Max?
maxout10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
maxout10/Reshape_1/shape?
maxout10/Reshape_1Reshapemaxout10/maxout_1/Max:output:0!maxout10/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
maxout10/Reshape_1?
BN11/add/ReadVariableOpReadVariableOp bn11_add_readvariableop_resource*
_output_shapes
:*
dtype02
BN11/add/ReadVariableOp?
BN11/addAddV2maxout10/Reshape_1:output:0BN11/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

BN11/add?
BN11/MatMul/ReadVariableOpReadVariableOp#bn11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
BN11/MatMul/ReadVariableOp?
BN11/MatMulMatMulmaxout10/Reshape_1:output:0"BN11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
BN11/MatMul?
dense12/MatMul/ReadVariableOpReadVariableOp&dense12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense12/MatMul/ReadVariableOp?
dense12/MatMulMatMulBN11/MatMul:product:0%dense12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense12/MatMul?
dense12/BiasAdd/ReadVariableOpReadVariableOp'dense12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense12/BiasAdd/ReadVariableOp?
dense12/BiasAddBiasAdddense12/MatMul:product:0&dense12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense12/BiasAddp
dense12/ReluReludense12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense12/Relu?
BN13/add/ReadVariableOpReadVariableOp bn13_add_readvariableop_resource*
_output_shapes
:*
dtype02
BN13/add/ReadVariableOp?
BN13/addAddV2dense12/Relu:activations:0BN13/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

BN13/add?
BN13/MatMul/ReadVariableOpReadVariableOp#bn13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
BN13/MatMul/ReadVariableOp?
BN13/MatMulMatMuldense12/Relu:activations:0"BN13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
BN13/MatMul?
dense14/MatMul/ReadVariableOpReadVariableOp&dense14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense14/MatMul/ReadVariableOp?
dense14/MatMulMatMulBN13/MatMul:product:0%dense14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense14/MatMul?
dense14/BiasAdd/ReadVariableOpReadVariableOp'dense14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense14/BiasAdd/ReadVariableOp?
dense14/BiasAddBiasAdddense14/MatMul:product:0&dense14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense14/BiasAddp
dense14/ReluReludense14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense14/Relu?
BN15/add/ReadVariableOpReadVariableOp bn15_add_readvariableop_resource*
_output_shapes
:*
dtype02
BN15/add/ReadVariableOp?
BN15/addAddV2dense14/Relu:activations:0BN15/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

BN15/add?
BN15/MatMul/ReadVariableOpReadVariableOp#bn15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
BN15/MatMul/ReadVariableOp?
BN15/MatMulMatMuldense14/Relu:activations:0"BN15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
BN15/MatMul?
dense16/MatMul/ReadVariableOpReadVariableOp&dense16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense16/MatMul/ReadVariableOp?
dense16/MatMulMatMulBN15/MatMul:product:0%dense16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense16/MatMul?
dense16/BiasAdd/ReadVariableOpReadVariableOp'dense16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense16/BiasAdd/ReadVariableOp?
dense16/BiasAddBiasAdddense16/MatMul:product:0&dense16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense16/BiasAddy
dense16/SoftmaxSoftmaxdense16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense16/Softmax?
IdentityIdentitydense16/Softmax:softmax:0^BN1/MatMul/ReadVariableOp^BN1/add/ReadVariableOp^BN11/MatMul/ReadVariableOp^BN11/add/ReadVariableOp^BN13/MatMul/ReadVariableOp^BN13/add/ReadVariableOp^BN15/MatMul/ReadVariableOp^BN15/add/ReadVariableOp^BN3/MatMul/ReadVariableOp^BN3/add/ReadVariableOp^BN5/MatMul/ReadVariableOp^BN5/add/ReadVariableOp^BN7/MatMul/ReadVariableOp^BN7/add/ReadVariableOp^BN9/MatMul/ReadVariableOp^BN9/add/ReadVariableOp^dense12/BiasAdd/ReadVariableOp^dense12/MatMul/ReadVariableOp^dense14/BiasAdd/ReadVariableOp^dense14/MatMul/ReadVariableOp^dense16/BiasAdd/ReadVariableOp^dense16/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^dense4/BiasAdd/ReadVariableOp^dense4/MatMul/ReadVariableOp^dense6/BiasAdd/ReadVariableOp^dense6/MatMul/ReadVariableOp^dense8/BiasAdd/ReadVariableOp^dense8/MatMul/ReadVariableOp^maxout0/MatMul/ReadVariableOp^maxout0/add/ReadVariableOp^maxout10/MatMul/ReadVariableOp^maxout10/add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????)::::::::::::::::::::::::::::::::::26
BN1/MatMul/ReadVariableOpBN1/MatMul/ReadVariableOp20
BN1/add/ReadVariableOpBN1/add/ReadVariableOp28
BN11/MatMul/ReadVariableOpBN11/MatMul/ReadVariableOp22
BN11/add/ReadVariableOpBN11/add/ReadVariableOp28
BN13/MatMul/ReadVariableOpBN13/MatMul/ReadVariableOp22
BN13/add/ReadVariableOpBN13/add/ReadVariableOp28
BN15/MatMul/ReadVariableOpBN15/MatMul/ReadVariableOp22
BN15/add/ReadVariableOpBN15/add/ReadVariableOp26
BN3/MatMul/ReadVariableOpBN3/MatMul/ReadVariableOp20
BN3/add/ReadVariableOpBN3/add/ReadVariableOp26
BN5/MatMul/ReadVariableOpBN5/MatMul/ReadVariableOp20
BN5/add/ReadVariableOpBN5/add/ReadVariableOp26
BN7/MatMul/ReadVariableOpBN7/MatMul/ReadVariableOp20
BN7/add/ReadVariableOpBN7/add/ReadVariableOp26
BN9/MatMul/ReadVariableOpBN9/MatMul/ReadVariableOp20
BN9/add/ReadVariableOpBN9/add/ReadVariableOp2@
dense12/BiasAdd/ReadVariableOpdense12/BiasAdd/ReadVariableOp2>
dense12/MatMul/ReadVariableOpdense12/MatMul/ReadVariableOp2@
dense14/BiasAdd/ReadVariableOpdense14/BiasAdd/ReadVariableOp2>
dense14/MatMul/ReadVariableOpdense14/MatMul/ReadVariableOp2@
dense16/BiasAdd/ReadVariableOpdense16/BiasAdd/ReadVariableOp2>
dense16/MatMul/ReadVariableOpdense16/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2>
dense4/BiasAdd/ReadVariableOpdense4/BiasAdd/ReadVariableOp2<
dense4/MatMul/ReadVariableOpdense4/MatMul/ReadVariableOp2>
dense6/BiasAdd/ReadVariableOpdense6/BiasAdd/ReadVariableOp2<
dense6/MatMul/ReadVariableOpdense6/MatMul/ReadVariableOp2>
dense8/BiasAdd/ReadVariableOpdense8/BiasAdd/ReadVariableOp2<
dense8/MatMul/ReadVariableOpdense8/MatMul/ReadVariableOp2>
maxout0/MatMul/ReadVariableOpmaxout0/MatMul/ReadVariableOp28
maxout0/add/ReadVariableOpmaxout0/add/ReadVariableOp2@
maxout10/MatMul/ReadVariableOpmaxout10/MatMul/ReadVariableOp2:
maxout10/add/ReadVariableOpmaxout10/add/ReadVariableOp:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
?
>__inference_BN11_layer_call_and_return_conditional_losses_2347

inputs
add_readvariableop_resource"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpi
addAddV2inputsadd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_dense14_layer_call_fn_2415

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense14_layer_call_and_return_conditional_losses_10972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input.
serving_default_input:0?????????);
dense160
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?a
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
layer_with_weights-16
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?Z
_tf_keras_network?Y{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 41]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Maxout1D", "config": {"units": 25, "output_units": 72}, "name": "maxout0", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN1", "trainable": true, "dtype": "float32"}, "name": "BN1", "inbound_nodes": [[["maxout0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 57, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["BN1", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN3", "trainable": true, "dtype": "float32"}, "name": "BN3", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense4", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense4", "inbound_nodes": [[["BN3", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN5", "trainable": true, "dtype": "float32"}, "name": "BN5", "inbound_nodes": [[["dense4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense6", "trainable": true, "dtype": "float32", "units": 48, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense6", "inbound_nodes": [[["BN5", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN7", "trainable": true, "dtype": "float32"}, "name": "BN7", "inbound_nodes": [[["dense6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense8", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense8", "inbound_nodes": [[["BN7", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN9", "trainable": true, "dtype": "float32"}, "name": "BN9", "inbound_nodes": [[["dense8", 0, 0, {}]]]}, {"class_name": "Maxout1D", "config": {"units": 25, "output_units": 24}, "name": "maxout10", "inbound_nodes": [[["BN9", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN11", "trainable": true, "dtype": "float32"}, "name": "BN11", "inbound_nodes": [[["maxout10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense12", "trainable": true, "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense12", "inbound_nodes": [[["BN11", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN13", "trainable": true, "dtype": "float32"}, "name": "BN13", "inbound_nodes": [[["dense12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense14", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense14", "inbound_nodes": [[["BN13", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN15", "trainable": true, "dtype": "float32"}, "name": "BN15", "inbound_nodes": [[["dense14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense16", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense16", "inbound_nodes": [[["BN15", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense16", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 41]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 41]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 41]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Maxout1D", "config": {"units": 25, "output_units": 72}, "name": "maxout0", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN1", "trainable": true, "dtype": "float32"}, "name": "BN1", "inbound_nodes": [[["maxout0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 57, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["BN1", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN3", "trainable": true, "dtype": "float32"}, "name": "BN3", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense4", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense4", "inbound_nodes": [[["BN3", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN5", "trainable": true, "dtype": "float32"}, "name": "BN5", "inbound_nodes": [[["dense4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense6", "trainable": true, "dtype": "float32", "units": 48, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense6", "inbound_nodes": [[["BN5", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN7", "trainable": true, "dtype": "float32"}, "name": "BN7", "inbound_nodes": [[["dense6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense8", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense8", "inbound_nodes": [[["BN7", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN9", "trainable": true, "dtype": "float32"}, "name": "BN9", "inbound_nodes": [[["dense8", 0, 0, {}]]]}, {"class_name": "Maxout1D", "config": {"units": 25, "output_units": 24}, "name": "maxout10", "inbound_nodes": [[["BN9", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN11", "trainable": true, "dtype": "float32"}, "name": "BN11", "inbound_nodes": [[["maxout10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense12", "trainable": true, "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense12", "inbound_nodes": [[["BN11", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN13", "trainable": true, "dtype": "float32"}, "name": "BN13", "inbound_nodes": [[["dense12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense14", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense14", "inbound_nodes": [[["BN13", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "BN15", "trainable": true, "dtype": "float32"}, "name": "BN15", "inbound_nodes": [[["dense14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense16", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense16", "inbound_nodes": [[["BN15", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense16", 0, 0]]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.005, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 41]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 41]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?
maxout_w
w
maxout_b
b
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Maxout1D", "name": "maxout0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"units": 25, "output_units": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 41]}}
?

norm_w
w

 norm_b
 b
!	variables
"trainable_variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalization", "name": "BN1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "BN1", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72]}}
?

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 57, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 72}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72]}}
?

+norm_w
+w

,norm_b
,b
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalization", "name": "BN3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "BN3", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 57]}}
?

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense4", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 57}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 57]}}
?

7norm_w
7w

8norm_b
8b
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalization", "name": "BN5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "BN5", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense6", "trainable": true, "dtype": "float32", "units": 48, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?

Cnorm_w
Cw

Dnorm_b
Db
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalization", "name": "BN7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "BN7", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48]}}
?

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense8", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48]}}
?

Onorm_w
Ow

Pnorm_b
Pb
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalization", "name": "BN9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "BN9", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
?
Umaxout_w
Uw
Vmaxout_b
Vb
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Maxout1D", "name": "maxout10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"units": 25, "output_units": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
?

[norm_w
[w

\norm_b
\b
]	variables
^trainable_variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalization", "name": "BN11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "BN11", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
?

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense12", "trainable": true, "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
?

gnorm_w
gw

hnorm_b
hb
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalization", "name": "BN13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "BN13", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
?

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense14", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
?

snorm_w
sw

tnorm_b
tb
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalization", "name": "BN15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "BN15", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
?

ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense16", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
"
	optimizer
?
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
I16
J17
O18
P19
U20
V21
[22
\23
a24
b25
g26
h27
m28
n29
s30
t31
y32
z33"
trackable_list_wrapper
?
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
I16
J17
O18
P19
U20
V21
[22
\23
a24
b25
g26
h27
m28
n29
s30
t31
y32
z33"
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
	variables
?metrics
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:!	)?2maxout0/maxout_w
:?2maxout0/maxout_b
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
	variables
?metrics
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:HH2
BN1/norm_w
:H2
BN1/norm_b
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
!	variables
?metrics
"trainable_variables
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:H92dense2/kernel
:92dense2/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
'	variables
?metrics
(trainable_variables
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:992
BN3/norm_w
:92
BN3/norm_b
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
-	variables
?metrics
.trainable_variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:9<2dense4/kernel
:<2dense4/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
3	variables
?metrics
4trainable_variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:<<2
BN5/norm_w
:<2
BN5/norm_b
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
9	variables
?metrics
:trainable_variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:<02dense6/kernel
:02dense6/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
?	variables
?metrics
@trainable_variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:002
BN7/norm_w
:02
BN7/norm_b
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
E	variables
?metrics
Ftrainable_variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:0$2dense8/kernel
:$2dense8/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
K	variables
?metrics
Ltrainable_variables
Mregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:$$2
BN9/norm_w
:$2
BN9/norm_b
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
Q	variables
?metrics
Rtrainable_variables
Sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"	$?2maxout10/maxout_w
 :?2maxout10/maxout_b
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
W	variables
?metrics
Xtrainable_variables
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2BN11/norm_w
:2BN11/norm_b
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
]	variables
?metrics
^trainable_variables
_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense12/kernel
:2dense12/bias
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
c	variables
?metrics
dtrainable_variables
eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2BN13/norm_w
:2BN13/norm_b
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
i	variables
?metrics
jtrainable_variables
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense14/kernel
:2dense14/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
o	variables
?metrics
ptrainable_variables
qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2BN15/norm_w
:2BN15/norm_b
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
u	variables
?metrics
vtrainable_variables
wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense16/kernel
:2dense16/bias
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
{	variables
?metrics
|trainable_variables
}regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
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
?2?
$__inference_model_layer_call_fn_2019
$__inference_model_layer_call_fn_1419
$__inference_model_layer_call_fn_2092
$__inference_model_layer_call_fn_1581?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_681?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *$?!
?
input?????????)
?2?
?__inference_model_layer_call_and_return_conditional_losses_1801
?__inference_model_layer_call_and_return_conditional_losses_1946
?__inference_model_layer_call_and_return_conditional_losses_1167
?__inference_model_layer_call_and_return_conditional_losses_1256?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_maxout0_layer_call_fn_2127?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_maxout0_layer_call_and_return_conditional_losses_2118?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
"__inference_BN1_layer_call_fn_2146?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
=__inference_BN1_layer_call_and_return_conditional_losses_2137?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense2_layer_call_fn_2166?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense2_layer_call_and_return_conditional_losses_2157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
"__inference_BN3_layer_call_fn_2185?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
=__inference_BN3_layer_call_and_return_conditional_losses_2176?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense4_layer_call_fn_2205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense4_layer_call_and_return_conditional_losses_2196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
"__inference_BN5_layer_call_fn_2224?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
=__inference_BN5_layer_call_and_return_conditional_losses_2215?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense6_layer_call_fn_2244?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense6_layer_call_and_return_conditional_losses_2235?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
"__inference_BN7_layer_call_fn_2263?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
=__inference_BN7_layer_call_and_return_conditional_losses_2254?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense8_layer_call_fn_2283?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense8_layer_call_and_return_conditional_losses_2274?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
"__inference_BN9_layer_call_fn_2302?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
=__inference_BN9_layer_call_and_return_conditional_losses_2293?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_maxout10_layer_call_fn_2337?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_maxout10_layer_call_and_return_conditional_losses_2328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_BN11_layer_call_fn_2356?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_BN11_layer_call_and_return_conditional_losses_2347?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense12_layer_call_fn_2376?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense12_layer_call_and_return_conditional_losses_2367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_BN13_layer_call_fn_2395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_BN13_layer_call_and_return_conditional_losses_2386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense14_layer_call_fn_2415?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense14_layer_call_and_return_conditional_losses_2406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_BN15_layer_call_fn_2434?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_BN15_layer_call_and_return_conditional_losses_2425?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense16_layer_call_fn_2454?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense16_layer_call_and_return_conditional_losses_2445?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_1656input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
>__inference_BN11_layer_call_and_return_conditional_losses_2347\\[/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
#__inference_BN11_layer_call_fn_2356O\[/?,
%?"
 ?
inputs?????????
? "???????????
>__inference_BN13_layer_call_and_return_conditional_losses_2386\hg/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
#__inference_BN13_layer_call_fn_2395Ohg/?,
%?"
 ?
inputs?????????
? "???????????
>__inference_BN15_layer_call_and_return_conditional_losses_2425\ts/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
#__inference_BN15_layer_call_fn_2434Ots/?,
%?"
 ?
inputs?????????
? "???????????
=__inference_BN1_layer_call_and_return_conditional_losses_2137\ /?,
%?"
 ?
inputs?????????H
? "%?"
?
0?????????H
? u
"__inference_BN1_layer_call_fn_2146O /?,
%?"
 ?
inputs?????????H
? "??????????H?
=__inference_BN3_layer_call_and_return_conditional_losses_2176\,+/?,
%?"
 ?
inputs?????????9
? "%?"
?
0?????????9
? u
"__inference_BN3_layer_call_fn_2185O,+/?,
%?"
 ?
inputs?????????9
? "??????????9?
=__inference_BN5_layer_call_and_return_conditional_losses_2215\87/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????<
? u
"__inference_BN5_layer_call_fn_2224O87/?,
%?"
 ?
inputs?????????<
? "??????????<?
=__inference_BN7_layer_call_and_return_conditional_losses_2254\DC/?,
%?"
 ?
inputs?????????0
? "%?"
?
0?????????0
? u
"__inference_BN7_layer_call_fn_2263ODC/?,
%?"
 ?
inputs?????????0
? "??????????0?
=__inference_BN9_layer_call_and_return_conditional_losses_2293\PO/?,
%?"
 ?
inputs?????????$
? "%?"
?
0?????????$
? u
"__inference_BN9_layer_call_fn_2302OPO/?,
%?"
 ?
inputs?????????$
? "??????????$?
__inference__wrapped_model_681?" %&,+1287=>DCIJPOUV\[abhgmntsyz.?+
$?!
?
input?????????)
? "1?.
,
dense16!?
dense16??????????
A__inference_dense12_layer_call_and_return_conditional_losses_2367\ab/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_dense12_layer_call_fn_2376Oab/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_dense14_layer_call_and_return_conditional_losses_2406\mn/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_dense14_layer_call_fn_2415Omn/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_dense16_layer_call_and_return_conditional_losses_2445\yz/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_dense16_layer_call_fn_2454Oyz/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense2_layer_call_and_return_conditional_losses_2157\%&/?,
%?"
 ?
inputs?????????H
? "%?"
?
0?????????9
? x
%__inference_dense2_layer_call_fn_2166O%&/?,
%?"
 ?
inputs?????????H
? "??????????9?
@__inference_dense4_layer_call_and_return_conditional_losses_2196\12/?,
%?"
 ?
inputs?????????9
? "%?"
?
0?????????<
? x
%__inference_dense4_layer_call_fn_2205O12/?,
%?"
 ?
inputs?????????9
? "??????????<?
@__inference_dense6_layer_call_and_return_conditional_losses_2235\=>/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????0
? x
%__inference_dense6_layer_call_fn_2244O=>/?,
%?"
 ?
inputs?????????<
? "??????????0?
@__inference_dense8_layer_call_and_return_conditional_losses_2274\IJ/?,
%?"
 ?
inputs?????????0
? "%?"
?
0?????????$
? x
%__inference_dense8_layer_call_fn_2283OIJ/?,
%?"
 ?
inputs?????????0
? "??????????$?
A__inference_maxout0_layer_call_and_return_conditional_losses_2118\/?,
%?"
 ?
inputs?????????)
? "%?"
?
0?????????H
? y
&__inference_maxout0_layer_call_fn_2127O/?,
%?"
 ?
inputs?????????)
? "??????????H?
B__inference_maxout10_layer_call_and_return_conditional_losses_2328\UV/?,
%?"
 ?
inputs?????????$
? "%?"
?
0?????????
? z
'__inference_maxout10_layer_call_fn_2337OUV/?,
%?"
 ?
inputs?????????$
? "???????????
?__inference_model_layer_call_and_return_conditional_losses_1167?" %&,+1287=>DCIJPOUV\[abhgmntsyz6?3
,?)
?
input?????????)
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_1256?" %&,+1287=>DCIJPOUV\[abhgmntsyz6?3
,?)
?
input?????????)
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_1801?" %&,+1287=>DCIJPOUV\[abhgmntsyz7?4
-?*
 ?
inputs?????????)
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_1946?" %&,+1287=>DCIJPOUV\[abhgmntsyz7?4
-?*
 ?
inputs?????????)
p 

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_1419v" %&,+1287=>DCIJPOUV\[abhgmntsyz6?3
,?)
?
input?????????)
p

 
? "???????????
$__inference_model_layer_call_fn_1581v" %&,+1287=>DCIJPOUV\[abhgmntsyz6?3
,?)
?
input?????????)
p 

 
? "???????????
$__inference_model_layer_call_fn_2019w" %&,+1287=>DCIJPOUV\[abhgmntsyz7?4
-?*
 ?
inputs?????????)
p

 
? "???????????
$__inference_model_layer_call_fn_2092w" %&,+1287=>DCIJPOUV\[abhgmntsyz7?4
-?*
 ?
inputs?????????)
p 

 
? "???????????
"__inference_signature_wrapper_1656?" %&,+1287=>DCIJPOUV\[abhgmntsyz7?4
? 
-?*
(
input?
input?????????)"1?.
,
dense16!?
dense16?????????