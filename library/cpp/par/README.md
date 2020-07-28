PARalyzer: library for distributed computations
===============================================

The library is intended for simple and effective use of multicore processors, as well as for distributed computing. It differs from Yandex mapreduce in absence of permanent storage (all data is stored only in memory) and in absence of fault tolerance. In a case of failure of one of hosts the program will most likely stop. As benefit, the PARalyzer provides very fast "transactions" (in terms of mapreduce) and automatically balances local and remove execution of elemental operations.

The simplest PARalyzer use case is fast execution of an operation for all elements of an array, which could be done by inheriting the operation from TMapReduceCmd<TInput,TOutput>, and then calling RunMap() function.

```cpp
inline void RunMap(IEnvironment *env, TMapReduceCmd<TInput,TOutput> *cmd, TVector<TInput> *src, TVector<TOutput> *dst)
```
To run the operation, it is required to provide:
1. IEnvironment (see below)
2. a command (inherited from TMapReduceCmd)
3. an input vector to perform the operation on
4. and a vector to store result

In order for an operation to run in distributed mode a serialisation ability is required for the object which implements the operation (it has to have serialising operator &, and a call to REGISTER_SAVELOAD_CLASS() must present). In addition to that, types of TInput and TOutput also must implement the serialisation operator, otherwise a POD-type serialisation will be applied. For serialisation details see mapreduce documentation.


Environment
-----------

IEnvironment allows transfer and storage of additional data. It is useful for case of several operation on the same data. For example, MatrixNet runs several iterations over the same set of features, which remains unchangeable over iterations. To save costs of data transfers such features could be stored into Environment for later reuse and thus transferred only once.

Class ```TCtxPtr<T>``` provides access to the data which was set thought Environment. Example of data transfer using Environment:
```cpp
void TMyOp::DoMap(IUserContext *ctx, int hostId, TInput *src, TOutput *dst) const
{
    TCtxPtr<TFeatures> pData(ctx, 0x12345, hostId);
    bla;
    bla;
}
...
void LaunchMyOp(IRootEnvironment *root, xz)
{
    TObj<TFeatures> pData = xz;
    TObj<IEnvironment> env = root->CreateEnvironment(0x12345, root->MakeHostIdMapping(1));
    env->SetContextData(0, pData);
    for (;;)
        RunMap(env, new TMyOp, src, dst);
}
```
Here 0x12345 is an unique id of Environment. In this example pData will be transferred only once, and then it can be reused infinite number of times.

For a case of data which doesn't fit memory of a single host it is possible to make Environment distributed. To split data into N parts it is necessary to provide IRootEnvironment::CreateEnvironment() with an array of mapping between number of worker and number of part. Such array could be created using root->MakeHostIdMapping(N) function. Here and below HostId is used as reference to a number of data part.

To release data memory on master host, if the data was shared using IEnvironment::SetContextData() call IEnvironment::MakeRemoteOnly(hostId).

When executing an operation on an Environment which is split into several parts, it is default that results of processing each part are merged into one common result using TMapReduceCmd::DoReduce() function. See TJobDescription for details.

To separate data which is updated often from data updated rarely a nested Environment could be created with help of IEnvironment::CreateChildEnvironment(). From inside of a child Environment it is possible to access its own data, as well as data of its predecessors. Mapping hostId of child environment will be same as mapping of its predecessor (otherwise it's impossible to perform operation on hosts with different hostId's). For example, in MatrixNet child environment is used to store data which is constant during one iteration: global Environment store examples with features (constant during the whole run), whereas child environment keeps current approximation of target function (updated each iteration).

For the case when global data is not required there are two fake Environment's: IRootEnvironment::GetAnywhere() and IRootEnvironment::GetEverywhere(). "Anywhere" operation could be executed at any node, "Everywhere" operation must run on each node and results has to be merged.


TJobDescription
---------------

Additionally (except RunMain()) there is RunMapReduce() function, which returns a single result combining results of separate TMapReduceCmd::DoMap() with use of TMapReduceCmd::DoReduce().

If features of RunMap() / RunMapReduce() are not enough, it is also possible to use descriptor TJopDescription of set of operations to add required commands directly. Array TJobDescription::ExecList[] lists elemental operations, which forms the set. Each operation has following properties:
1. CmdId - number of the operation in array TJobDescription::Cmds[]
2. ParamId - number of parameter in array of parameters TJobDescription::Params[]
3. ReduceId - id for result reduction. If several consecutive operations has equal ReduceId, then results of these will be merged using IDistrCmd::MergeAsync()
4. HostId - for distributed execution it is possible to run a command on a particular hostId or on a subset of hostId. There are also special values: ANYWHERE_HOST_ID - means that an operation could run on any host, MAP_HOST_ID - means that operation must run on all hostId and result must be merged using IDistrCmd::MergeAsync().
5. Additionally there is CompId field which is filled on execution of a command with id of the current host.

For a more convenient setting TJobDescription there are functions present:
1. SetCurrentOperation() - sets the command which will be used. It is possible to provide the command itself, it will be serialised. Or it's possible to provide already serialised data.
2. AddMap() - adds an operation to run on all hostId.
3. AddQuery() - allows to specify on which hostId(s) to run.
4. MergeResults() - sets all ReduceId to the same value, which means that results of all commands to be merged.
5. SeparateResults() - converts all MAP_HOST_ID operation into separate lists of operations for each hostId.


Asynchronous execution of operations
------------------------------------

TMapReduceCmd<> is a wrapper. Interface IDistrCmd is used inside of PARalyzer. The interface uses asynchronous notification IDCResultNotify about operation completion. This allows implementation of commands which wait some event.

For example, RemoteMap uses child of IDistrCmd, which does distributed runs of a part of a task, and on completion sends a result through IDCResultNotify. Async notification allows to avoid separate thread/coroutine for each operation and avoid synchronisation primitives.  This simplifies debugging, avoids limits on stack size and just works faster.


RemoteMap/RemoteMapReduce
-------------------------

For the case when result of RunMap() is input for the next RunMap() it is possible to avoid transfer of data of first RunMap() to head node and run the second RunMap() in the location of results of the first one. This is implemented via RemoteMap() and RemoteMapReduce(). Example:

```cpp
void EvaluateModels(IEnvironment *env, const vector<TModel> &models, vector<float> *res)
{
    TJobDescription jd;
    Map(&jd, new TScoreCalcer, &models);
    RemoteMap(&jd, new TFinalScoreCalcer);
    TJobExecutor req(&jd, env);
    req.GetRemoteMapResults(res);
}
```

In RemoteMap() the source task of N operations will be split into Min(100, sqrt(N)) parts, thus it make sense to use RemoteMap() for cases when number of results of the first Map() is huge.


Location of execution for an operation
--------------------------------------

In distributed mode one operation could be executed on several hosts. Result of the one finished first is used. For example elemental operations run simultaneously on head node and remote node (local-remote balance). This allows to pick minimal time between "transfer + remote run" and "local run". Additionally operations which run on the slowest host will also be executed somewhere else. This allows for case of unexpected slowdown of a host to keep overall execution time increase not more than 2 times.


Initialisation/termination
--------------------------

Following functions are responsible for PARalyzer initialisation:
```cpp
void RunSlave(int workerThreadCount, int port);
IRootEnvironment *RunMaster(int workerThreadCount, const char *hostsFileName, int port);
IRootEnvironment *RunLocalMaster(int workerThreadCount);
```

RunSlave() - starts a working node.
RunMaster() - starts head node, with list of working nodes from hostsFileName.
RunLocalMaster() - start head node locally, with no network and distributed capabilities.
Port - ip port number for data communication
WorkerThreadCount - number of worker threads, for an optimal performance it has to be less than or equal to number of physical cores, otherwise latency may increase due context switches and waiting for inactive threads.

