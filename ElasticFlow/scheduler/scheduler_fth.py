from __future__ import print_function
from collections import OrderedDict
import copy
import csv
import cvxpy as cp
import re
import sys
import types
import time
import threading
import math
#parse args
import argparse
import copy
import os

import utils
import flags
import jobs
import cluster
import log
#import lp
import profiles
from runtime.rpc import scheduler_client
from runtime.rpc import scheduler_server

# import hosts
# import placement_scheme as scheme
# import cmd

#parse input arguments
# flags.DEFINE_string(name, default, help_string, flag_values=_flagvalues.FLAGS)
# name：字符串，表示命令行参数的名称。
# default：字符串，表示参数的默认值。如果在命令行中未提供该参数，则使用此默认值。
# help_string：字符串，描述参数的用途，当使用命令行帮助时显示。
# flag_values：可选，指定一个 flags.FlagValues() 对象。如果不指定，它将使用默认的全局 FLAGS 对象。
flags.DEFINE_string('trace_file', 'tf_job.csv',
                '''Provide TF job trace file 提供TF作业trace文件(*.csv, *.txt).
                    *.csv file, use \',\' as delimiter; *.txt file, user \' \' as deliminter.  *.csv文件使用','作为分隔符；*.txt文件使用' '作为分隔符。
                    Default file is tf_job.csv ''')
flags.DEFINE_string('log_path', 'result-' + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
                '''Simulation output folder, including cluster/node/gpu usage trace, pending job_queue info. 模拟输出文件夹，包括集群/节点/ GPU 使用情况追踪，挂起作业队列信息。
                Default folder is result-[time]''')
flags.DEFINE_string('scheme', 'yarn',
                '''
                Job placement scheme: 作业放置方案：
                0.count, just resource counting, without assignment (which gpu, which cpu) 计数，仅资源计数，不包括分配（哪个GPU，哪个CPU）
                1.yarn, ms yarn 
                2.random 随机
                3.crandom (consolidate + random) （合并 + 随机）
                4.greedy
                5.balance
                6.cbalance (consolidate + balance) （合并 + 平衡）
                Default is yarn''')
flags.DEFINE_string('schedule', 'fifo',  
                '''
                Job schedule scheme: 作业调度方案：
                1.fifo
                2.fjf, fit job first( in fifo order)
                3.sjf, smallest job first
                4.lpjf, longest pending job first 最长挂起作业优先
                5.shortest, shortest-remaining-time job first 剩余时间最短的作业优先
                6.shortest-gpu, shortest-remaining-gputime job first  剩余GPU时间最短的作业优先
                7.dlas, discretized las   分段线性加速调度 
                8.dlas-gpu, dlas using gpu time 使用GPU时间的DLAS
                Default is fifo''')
flags.DEFINE_integer('num_switch', 1, 
                '''Part of cluster spec: the number of switches in this cluster, default is 1 集群规格的一部分：此集群中的交换机数量，默认为1''')
flags.DEFINE_integer('num_node_p_switch', 32, 
                '''Part of cluster spec: the number of nodes under a single switch, default is 32''')
flags.DEFINE_integer('num_gpu_p_node', 8, 
                '''Part of cluster spec: the number of gpus on each node, default is 8''')
flags.DEFINE_integer('num_cpu_p_node', 64,
                '''Part of cluster spec: the number of cpus on each node, default is 64''')
flags.DEFINE_integer('mem_p_node', 256,
                '''Part of cluster spec: memory capacity on each node, default is 128''')
flags.DEFINE_integer('scheduling_slot', 60,
                '''The least re-scheduling time slot for ef and edf EF和EDF的最小重新调度时间槽''')
flags.DEFINE_integer('restart_threshold', 100,
                '''restart trainers after a while 一段时间后重新启动训练器''')
flags.DEFINE_string('cluster_spec', None,
                '''Part of cluster spec: cluster infra spec file, 
                this file will overwrite the specs from num_switch, num_node_p_switch, and num_gpu_p_node
                Spec format:
                    num_switch,num_node_p_switch,num_gpu_p_node
                    int,int,int''')

flags.DEFINE_boolean('print', False, 
                '''Enable print out information, default is False 启用信息打印，默认为False''')
flags.DEFINE_boolean('flush_stdout', True, 
                '''Flush stdout, default is True 刷新stdout，默认为True''')
flags.DEFINE_boolean('simulation', True, 
                '''whether the scheduler is for simulation or phisical cluster experiments, default is True 调度程序是否用于模拟或物理集群实验，默认为True' ''')
flags.DEFINE_boolean('fastforward', True, 
                '''Whether to fastforward the cluster experiments process, default is True 是否快进集群实验过程，默认为True ''')
flags.DEFINE_version('0.1')
flags.DEFINE_boolean('early_stop', False, 
                '''Whether to stop a job if a target metric is reached 是否在达到目标指标时停止作业 ''')
flags.DEFINE_string('gpu_type', 'A100', 
                '''The GPU to run on. It should match the trace provided. 要运行的GPU。它应与提供的跟踪匹配。''')
flags.DEFINE_boolean('plot_figure', True, 
                '''Whether to write log files and plot figures afterwards''')


FLAGS = flags.FLAGS

# ############
# # 解析输入参数
# flags.DEFINE_string('trace_file', 'tf_job.csv',
#                 '''提供TF作业追踪文件（*.csv, *.txt）。
#                     *.csv文件使用','作为分隔符；*.txt文件使用' '作为分隔符。
#                     默认文件是tf_job.csv ''')
# flags.DEFINE_string('log_path', 'result-' + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
#                 '''模拟输出文件夹，包括集群/节点/ GPU 使用情况追踪，挂起作业队列信息。
#                 默认文件夹是result-[时间]''')
# flags.DEFINE_string('scheme', 'yarn',
#                 '''
#                 作业放置方案：
#                 0.计数，仅资源计数，不包括分配（哪个GPU，哪个CPU）
#                 1.yarn，ms yarn
#                 2.随机
#                 3.crandom（合并 + 随机）
#                 4.贪婪
#                 5.平衡
#                 6.cbalance（合并 + 平衡）
#                 默认是yarn''')
# flags.DEFINE_string('schedule', 'fifo',
#                 '''
#                 作业调度方案：
#                 1.fifo
#                 2.fjf，先适应作业（按fifo顺序）
#                 3.sjf，最小作业优先
#                 4.lpjf，最长挂起作业优先
#                 5.shortest，剩余时间最短的作业优先
#                 6.shortest-gpu，剩余GPU时间最短的作业优先
#                 7.dlas，分段线性加速调度
#                 8.dlas-gpu，使用GPU时间的DLAS
#                 默认是fifo''')
# flags.DEFINE_integer('num_switch', 1, 
#                 '''集群规格的一部分：此集群中的交换机数量，默认为1''')
# flags.DEFINE_integer('num_node_p_switch', 32, 
#                 '''集群规格的一部分：单个交换机下的节点数，默认为32''')
# flags.DEFINE_integer('num_gpu_p_node', 8, 
#                 '''集群规格的一部分：每个节点上的GPU数量，默认为8''')
# flags.DEFINE_integer('num_cpu_p_node', 64,
#                 '''集群规格的一部分：每个节点上的CPU数量，默认为64''')
# flags.DEFINE_integer('mem_p_node', 256,
#                 '''集群规格的一部分：每个节点的内存容量，默认为128''')
# flags.DEFINE_integer('scheduling_slot', 60,
#                 '''EF和EDF的最小重新调度时间槽''')
# flags.DEFINE_integer('restart_threshold', 100,
#                 '''一段时间后重新启动训练器''')
# flags.DEFINE_string('cluster_spec', None,
#                 '''集群基础设施规格文件，
#                 此文件将覆盖num_switch，num_node_p_switch和num_gpu_p_node中的规格
#                 规格格式：
#                     num_switch,num_node_p_switch,num_gpu_p_node
#                     int,int,int''')

# flags.DEFINE_boolean('print', False, 
#                 '''启用信息打印，默认为False''')
# flags.DEFINE_boolean('flush_stdout', True, 
#                 '''刷新stdout，默认为True''')
# flags.DEFINE_boolean('simulation', True, 
#                 '''调度程序是否用于模拟或物理集群实验，默认为True''')
# flags.DEFINE_boolean('fastforward', True, 
#                 '''是否快进集群实验过程，默认为True''')
# flags.DEFINE_version('0.1')
# flags.DEFINE_boolean('early_stop', False, 
#                 '''是否在达到目标指标时停止作业''')
# flags.DEFINE_string('gpu_type', 'A100', 
#                 '''要运行的GPU。它应与提供的跟踪匹配。''')
# flags.DEFINE_boolean('plot_figure', True, 
#                 '''是否在之后写入日志文件并绘制图形''')


##########



#prepare JOBS list
JOBS = jobs.JOBS

#get host info
CLUSTER = cluster.CLUSTER

#get LOG object
LOG = log.LOG

#pre-run throughput information
THROUGHPUTS = profiles.THROUGHPUTS

job_stable = dict()
fast_forward_permission = False

MASTER_PORT = 22224
last_round_running_jobs, this_round_running_jobs = dict(), dict()
trainers_to_kill = {}
this_round_begin_time = None
last_round_gpu_allocations, gpu_allocations = None, None
job_to_be_killed = False
# for dlas-gpu cluster experiments
run_jobs = None

global_lock = threading.Lock()
global_ready_lock = threading.Lock()
commands = []

schedule_count = 0

def report_ready_callback(trainer_id):
    """Callback for tainers reporting ready. For overhead estimation and for debug.
    @param trainer_id: The id of the ready trainer.
    """
    print("received report ready request of trainer_id", trainer_id)
    global trainers_to_kill, global_ready_lock, this_round_begin_time
    global job_stable, commands, job_to_be_killed, last_round_gpu_allocations
    global gpu_allocations
    global_ready_lock.acquire()
    """for eachjob in trainers_to_kill:
        if trainer_id in trainers_to_kill[eachjob]:
            trainers_to_kill[eachjob].remove(trainer_id)
    for eachjob in trainers_to_kill:
        if len(trainers_to_kill[eachjob]) > 0:
            global_ready_lock.release()
            return"""
    last_round_gpu_allocations[trainer_id // CLUSTER.num_gpu_p_node][trainer_id % CLUSTER.num_gpu_p_node] = 0
    for node in last_round_gpu_allocations:
        for gpu in node:
            if gpu == 1:
                global_ready_lock.release()
                return
    for command in commands:
        scheduler_rpc_client.schedule(command)
    scheduler_rpc_client.schedule('F')
    scheduler_rpc_client.schedule('T')
    # all jobs have been killed. no running jobs in cluster
    job_to_be_killed = False
    last_round_gpu_allocations = gpu_allocations
    global_ready_lock.release()


def report_stable_callback(job_id):
    """Callback for tainers reporting stable status and prepare for fast forward
    @param job_id: The id of the stable job(s).
    """
    print("received fastforward request of job", job_id)
    receive_time = time.time()
    global job_stable, fast_forward_permission, this_round_begin_time, global_lock
    global_lock.acquire()
    if job_id not in job_stable:
        print("job", job_id, "requested fast forward before scaling")
    if job_stable[job_id] != 0:
        print("unexpected request from job", job_id, job_stable)
    assert job_id in job_stable and job_stable[job_id] == 0
    job_stable[job_id] = 1
    """if FLAGS.schedule == 'dlas-gpu':
        # workaround
        job = utils.search_dict_list(JOBS.runnable_jobs, 'job_idx', job_id)
        job['overhead'] = math.floor(receive_time - this_round_begin_time)
        print("job", job_id, "overhead", job['overhead'])
        for each_job in JOBS.runnable_jobs:
            if each_job['status'] != 'RUNNING':
                continue
            if each_job['num_gpu'] == 0 or each_job['placements'] is None or len(each_job['placements']) == 0:
                continue
            if each_job['job_idx'] not in job_stable:
                if each_job['job_idx'] in this_round_running_jobs:
                    each_job['overhead'] = 0
                    continue
                global_lock.release()
                return
            if job_stable[each_job['job_idx']] == 0:
                global_lock.release()
                return"""
    if FLAGS.schedule == 'dlas-gpu':
        job = utils.search_dict_list(JOBS.runnable_jobs, 'job_idx', job_id)
        job['overhead'] = math.floor(receive_time - this_round_begin_time)
        print("job", job_id, "overhead", job['overhead'])
        if job['overhead'] > 20:
            job['overhead'] = 20
        for each_job in JOBS.runnable_jobs:
            if each_job['status'] != 'RUNNING':
                continue
            if each_job['num_gpu'] == 0 or each_job['node_set'] is None:
                continue
            if each_job['job_idx'] not in job_stable:
                if each_job['job_idx'] in this_round_running_jobs:
                    each_job['overhead'] = 0
                    continue
                global_lock.release()
                return
            if job_stable[each_job['job_idx']] == 0:
                global_lock.release()
                return
    else:
        job = utils.search_dict_list(JOBS.running_jobs, 'job_idx', job_id)
        job['overhead'] = math.floor(receive_time - this_round_begin_time)
        print("job", job_id, "overhead", job['overhead'])
        if job['overhead'] > 20:
            job['overhead'] = 20
        for each_job in JOBS.running_jobs:
            if each_job['num_gpu'] == 0 or each_job['node_set'] is None:
                continue
            if each_job['job_idx'] not in job_stable:
                if each_job['job_idx'] in this_round_running_jobs:
                    each_job['overhead'] = 0
                    continue
                global_lock.release()
                return
            if job_stable[each_job['job_idx']] == 0:
                global_lock.release()
                return
    fast_forward_permission = True
    global_lock.release()
    print("ALL JOBS READY")


def parse_job_file(trace_file):
    #check trace_file is *.csv
    fd = open(trace_file, 'r')
    deli = ','
    if ((trace_file.find('.csv') == (len(trace_file) - 4))):
        deli = ','
    elif ((trace_file.find('.txt') == (len(trace_file) - 4))):
        deli = ' '

    reader = csv.DictReader(fd, delimiter = deli) 
    ''' Add job from job trace file'''
    keys = reader.fieldnames
    utils.print_fn('--------------------------------- Read TF jobs from: %s ---------------------------------' % trace_file) 
    utils.print_fn('    we get the following fields:\n        %s' % keys)
    job_idx = 0
    for row in reader:
        #add job into JOBS
        JOBS.add_job(row)
        # JOBS.read_job_info(job_idx, 'num_gpu')
        job_idx += 1

    assert job_idx == len(JOBS.job_list) 
    assert JOBS.num_job == len(JOBS.job_list) 
    # JOBS.print_all_job_size_info()
    JOBS.sort_all_jobs()
    # print(lp.prepare_job_info(JOBS.job_list[0]))
    utils.print_fn('---------------------------------- Get %d TF jobs in total ----------------------------------' % job_idx)
    # JOBS.read_all_jobs()
    fd.close()

def parse_cluster_spec():
    global last_round_gpu_allocations
    if FLAGS.cluster_spec:
        print(FLAGS.cluster_spec)
        spec_file = FLAGS.cluster_spec
        fd = open(spec_file, 'r')
        deli = ','
        if ((spec_file.find('.csv') == (len(spec_file) - 4))):
            deli = ','
        elif ((spec_file.find('.txt') == (len(spec_file) - 4))):
            deli = ' '
        reader = csv.DictReader(fd, delimiter = deli) 
        keys = reader.fieldnames
        utils.print_fn(keys)
        if 'num_switch' not in keys:
            return
        if 'num_node_p_switch' not in keys:
            return
        if 'num_gpu_p_node' not in keys:
            return
        if 'num_cpu_p_node' not in keys:
            return
        if 'mem_p_node' not in keys:
            return
        
        ''' there should be only one line remaining'''
        assert reader.line_num == 1

        ''' get cluster spec '''
        for row in reader:
            # utils.print_fn('num_switch %s' % row['num_switch'])
            FLAGS.num_switch = int(row['num_switch'])
            FLAGS.num_node_p_switch = int(row['num_node_p_switch'])
            FLAGS.num_gpu_p_node = int(row['num_gpu_p_node'])
            FLAGS.num_cpu_p_node = int(row['num_cpu_p_node'])
            FLAGS.mem_p_node = int(row['mem_p_node'])
        fd.close()

    utils.print_fn("num_switch: %d" % FLAGS.num_switch)
    utils.print_fn("num_node_p_switch: %d" % FLAGS.num_node_p_switch)
    utils.print_fn("num_gpu_p_node: %d" % FLAGS.num_gpu_p_node)
    utils.print_fn("num_cpu_p_node: %d" % FLAGS.num_cpu_p_node)
    utils.print_fn("mem_p_node: %d" % FLAGS.mem_p_node)

    '''init infra'''
    CLUSTER.init_infra()
    # utils.print_fn(lp.prepare_cluster_info())
    last_round_gpu_allocations = [[0 for gpu in range(CLUSTER.num_gpu_p_node)] for _ in range(CLUSTER.num_node)]
    utils.print_fn('--------------------------------- End of cluster spec ---------------------------------')
    return 


'''
Allocate job resource
'''
def try_get_job_res(job):
    '''
    select placement scheme
    '''
    if FLAGS.scheme == 'elastic':
        ret = CLUSTER.elastic_placement(job)
    elif FLAGS.scheme == 'yarn':
        ret = CLUSTER.ms_yarn_placement(job)
    elif FLAGS.scheme == 'balance':
        ret = lp.placement(job)
        # ret = lp.min_new_job(job)
    elif FLAGS.scheme == 'random':
        ret = CLUSTER.random_placement(job)
    elif FLAGS.scheme == 'crandom':
        ret = CLUSTER.consolidate_random_placement(job)
    elif FLAGS.scheme == 'greedy':
        ret = CLUSTER.greedy_placement(job)
    elif FLAGS.scheme == 'gandiva':
        ret = CLUSTER.gandiva_placement(job)
    elif FLAGS.scheme == 'count':
        ret = CLUSTER.none_placement(job)
    else:
        ret = CLUSTER.ms_yarn_placement(job)
    if ret == True:
        # job['status'] = 'RUNNING'
        pass
    return ret


#ef_allocation_heuristic
# 这段代码是ElasticFlow（EF）调度算法的一部分，它负责模拟和分配作业的GPU资源。这个过程旨在确保作业能够在其截止时间（DDL）之前完成，同时尽可能高效地利用集群资源。下面是对代码的逐行详解：
# 函数ef_sim_allocation定义了EF算法的资源分配过程。它接收多个参数，包括作业字典job_dict、起始时间start_time、
# 是否分配GPUassign_gpu、已分配的GPU数量assigned_gpus、是否为模拟simulation以及未来空闲GPU的字典future_free_gpus
def ef_sim_allocation(job_dict, start_time, assign_gpu=False, assigned_gpus=-1, simulation=False, future_free_gpus=None):
    event_time = start_time # event_time设置为start_time，用于跟踪当前事件的时间。
    return_value = True
    global_batch_size = str(job_dict['batch_size'])
    if 'iter_left' not in job_dict:
        job_dict['iter_left'] = job_dict['iteration']
    iter_left = job_dict['iter_left']
    if iter_left <= 0:
        print(job_dict)
    assert iter_left > 0

    new_allocations = {event_time:0} # new_allocations字典用于存储新的GPU分配计划。
    if future_free_gpus is None:  # future_free_gpus是包含未来时间点上可用GPU数量的字典。如果没有提供，就使用CLUSTER.future_free_gpus。
        future_free_gpus = CLUSTER.future_free_gpus
    new_future_free_gpus = copy.deepcopy(future_free_gpus) # new_future_free_gpus是future_free_gpus的深拷贝，用于在函数中更新。

    # aligned_ddl是将作业的DDL时间对齐到调度槽的开始时间。
    aligned_ddl = utils.align_to_time_slot(start_time, job_dict['ddl'], FLAGS.scheduling_slot)
    # add ddl into future free gpus
    # 如果aligned_ddl不在new_future_free_gpus中，并且大于等于start_time，则将aligned_ddl添加到new_future_free_gpus字典中，并设置其值为最后一个事件时间的GPU数量。
    if aligned_ddl not in new_future_free_gpus and aligned_ddl >= start_time:
        for each_event_time in new_future_free_gpus:
            if each_event_time < aligned_ddl:
                last_event_gpu = new_future_free_gpus[each_event_time]
            else:
                break
        new_future_free_gpus[aligned_ddl] = last_event_gpu
        new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
    
    # available_gpu是start_time时刻可用的GPU数量。
    available_gpu = new_future_free_gpus[start_time]
    # 如果assign_gpu为True，则根据assigned_gpus分配GPU。quota是分配给作业的GPU数量。
    if assign_gpu:
        quota = assigned_gpus
        assert quota <= available_gpu
        found_future_time = False
        # 循环遍历future_free_gpus，找到第一个不小于event_time的未来事件时间future_event_time。
        for each_event_time in future_free_gpus:
            future_event_time = each_event_time
            if future_event_time < event_time:
                available_gpu = future_free_gpus[future_event_time]
                del new_future_free_gpus[future_event_time]
                continue
            elif future_event_time == event_time:
                available_gpu = future_free_gpus[future_event_time]
                continue
            found_future_time = True
            break
        if not found_future_time:
            future_event_time = aligned_ddl
        new_allocations[event_time] = quota
        # 计算作业在新分配下的预计完成时间estimated_real_end_time和estimated_end_time。
        estimated_real_end_time = math.ceil(event_time + iter_left / float(
            THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(quota)]))
        estimated_end_time = utils.align_to_time_slot(start_time, estimated_real_end_time, FLAGS.scheduling_slot)
        # 如果预计的未来事件时间future_event_time大于或等于预计的结束时间estimated_end_time且DDL大于或等于estimated_real_end_time，则认为作业可以在DDL之前完成。
        # 更新new_future_free_gpus，移除已经分配的GPU，并更新作业的结束时间。
        if future_event_time >= estimated_end_time and job_dict['ddl'] >= estimated_real_end_time:
            # 遍历 future_free_gpus 中未来时间点上的 GPU 数量，并根据预计结束时间 estimated_end_time 进行更新。
            for each_event_time in future_free_gpus:
                # 如果当前事件时间小于函数中记录的当前事件时间 event_time，则跳过该事件时间
                if each_event_time < event_time:
                    continue
                
                # 如果当前事件时间小于预计的结束时间 estimated_end_time，则记录当前事件时间点的 GPU 数量，并更新该时间点的 GPU 数量，减去本次分配的 GPU 数量 quota，确保 GPU 数量不会小于零。
                if each_event_time < estimated_end_time:
                    last_event_gpu = new_future_free_gpus[each_event_time]
                    new_future_free_gpus[each_event_time] -= quota
                    assert new_future_free_gpus[each_event_time] >= 0

            # 如果当前事件时间不等于预计的结束时间，更新当前事件时间点的 GPU 数量，使其等于可用 GPU 数量减去分配的 GPU 数量 quota，并确保 GPU 数量不会小于零。
            if event_time != estimated_end_time:
                new_future_free_gpus[event_time] = available_gpu - quota
            assert new_future_free_gpus[event_time] >= 0
            # 如果预计的结束时间不在未来可用 GPU 数量的字典 future_free_gpus 中，
            # 则将预计的结束时间添加到 new_future_free_gpus 中，并设置其 GPU 数量为最后一个记录的 GPU 数量 last_event_gpu，并确保 GPU 数量不会小于零。
            if estimated_end_time not in future_free_gpus:
                new_future_free_gpus[estimated_end_time] = last_event_gpu
                assert new_future_free_gpus[event_time] >= 0
            # 对更新后的 new_future_free_gpus 字典按照时间进行排序。
            new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
                
            # 如果simulation为True，则更新作业字典中的模拟相关字段。否则，更新集群的未来可用 GPU 数量和作业的结束时间，并将 GPU 分配信息更新到作业字典中。
            if simulation:
                job_dict['next_level_allocation'] = utils.merge_dict(new_allocations)
                job_dict['marginal_gputime'] = (utils.get_allocation_time(
                    new_allocations, estimated_end_time) - utils.get_allocation_time(
                    job_dict['allocations'], job_dict['end_time'])) / (job_dict['next_level_gpu'] - job_dict['num_gpu'])
                job_dict['next_level_future_gpus'] = utils.merge_dict(new_future_free_gpus)
                job_dict['next_level_endtime'] = estimated_end_time
                job_dict['next_level_realendtime'] = estimated_real_end_time
            else:
                if assign_gpu:
                    job_dict['old_end_time'] = job_dict['end_time']
                CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                job_dict['end_time'] = estimated_end_time
                job_dict['real_end_time'] = estimated_real_end_time
                job_dict['allocations'] = utils.merge_dict(new_allocations)
                
            # 返回是否作业可以在截止时间之前完成的布尔值。
            return estimated_real_end_time <= job_dict['ddl']
        # 如果预计的未来事件时间小于预计的结束时间，或者作业的截止时间小于预计的真实结束时间，则执行以下操作：
        else:
            # 计算当前事件时间 event_time 到未来事件时间 future_event_time 之间的时间间隔，并根据 GPU 的吞吐量计算此间隔内作业完成的迭代次数，并更新剩余迭代次数 iter_left。
            duration = future_event_time - event_time
            if future_event_time == aligned_ddl:
                duration = job_dict['ddl'] - event_time
            iterations = duration * float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                quota)])
            iter_left -= iterations
            # 更新当前事件时间点的 GPU 数量，以反映已分配的 GPU。
            if event_time in new_future_free_gpus:
                new_future_free_gpus[event_time] -= quota
            else:
                new_future_free_gpus[event_time] = last_event_gpu - new_allocations[event_time]
            assert new_future_free_gpus[event_time] >= 0
        # 将事件时间更新为下一个未来事件时间 future_event_time，并将此事件时间点的 GPU 分配数量设置为 0，以表示 GPU 的释放。
        event_time = future_event_time # assign GPU!
        new_allocations[event_time] = 0

    
    quota = utils.get_next_level(job_dict, gpu_num=0) # 获取下一级的 GPU 配额
    last_quota, last_last_quota = 0, 0 # 上一次和上上次的 GPU 配额初始化为 0
    point = start_time - 1 # 初始化 point 为开始时间的前一时刻
    # allocate from 1 gpu only
    while quota <= CLUSTER.num_gpu and quota <= job_dict['max_gpu']: # 只分配一个 GPU

        period_end = aligned_ddl # 初始化周期结束时间为对齐的截止时间
        tmp_iters = 0 # 初始化临时迭代次数为 0
        
        # 遍历新未来 GPU 自由量字典中的时间段
        for period_start in new_future_free_gpus:
            if period_start >= aligned_ddl:
                continue
            if period_start not in new_allocations:
                new_allocations[period_start] = 0
                
        # 反向遍历新未来 GPU 自由量字典中的时间段
        for period_start in reversed(list(new_future_free_gpus.keys())):
            if period_start >= aligned_ddl:
                continue
            if period_start < event_time:
                break
            allocation = quota
            
            # 如果当前时间段 GPU 数量小于分配的配额，更新分配的配额
            if new_future_free_gpus[period_start] < quota:
                allocation = min(quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[period_start]))
                
            # 如果分配为 0，则更新周期结束时间并继续下一次循环
            if allocation == 0:
                period_end = period_start
                continue
            # 计算当前时间段的迭代次数
            duration = period_end - period_start
            if period_end == aligned_ddl:
                duration = job_dict['ddl'] - period_start
            tmp_iters += duration * float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                    allocation)])
            
            # 如果迭代次数超过剩余迭代次数，更新 point，并根据 point 和 GPU 分配情况估计结束时间
            if tmp_iters >= iter_left:
                point = period_start
                # allocate from period start, and return! # 从时间段开始分配，并返回！
                # 以下代码块是用于计算在当前分配下，作业何时能够完成，并更新相关的分配信息
                last_allocation_time = None
                last_allocation = None
                estimated_end_time = None
                for eachtime in new_future_free_gpus:
                    if eachtime > aligned_ddl:
                        break # >= # 如果超过对齐的截止时间，则结束循环
                    if eachtime < event_time:
                        continue # 如果时间小于事件时间，则继续下一次循环
                    
                    if last_allocation_time is None:
                        last_allocation_time = eachtime
                        # 获取最后一个级别的 GPU 数量
                        if eachtime >= period_start:
                            last_allocation = min(quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                        else:
                            last_allocation = min(last_quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                        continue
                    # 更新分配情况和未来 GPU 自由量    
                    new_allocations[last_allocation_time] = last_allocation
                    new_future_free_gpus[last_allocation_time] -= new_allocations[last_allocation_time]
                    assert new_future_free_gpus[last_allocation_time] >= 0

                    if last_allocation == 0:
                        last_allocation_time = eachtime
                        # 获取最后一个级别的 GPU 数量
                        if eachtime >= period_start:
                            last_allocation = min(quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                        else:
                            last_allocation = min(last_quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                        continue

                    duration = eachtime - last_allocation_time
                    if eachtime == aligned_ddl:
                        duration = job_dict['ddl'] - last_allocation_time
                    # 计算迭代次数
                    iterations = duration * float(
                        THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                        last_allocation)])
                    if last_allocation_time < period_start:
                        assert iterations < iter_left
                        iter_left -= iterations
                    else:
                        if iterations < iter_left:
                            iter_left -= iterations
                        else:
                            # 根据剩余迭代次数和 GPU 吞吐量估算结束时间
                            real_time_left = math.ceil(iter_left / float(
                                THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                                new_allocations[last_allocation_time])]))
                            time_left = utils.align_to_time_slot(0, real_time_left, FLAGS.scheduling_slot)
                            estimated_end_time = last_allocation_time + time_left
                            estimated_real_end_time = last_allocation_time + real_time_left
                            if estimated_end_time > eachtime:
                                print(estimated_end_time, eachtime, new_future_free_gpus)
                            assert estimated_end_time <= eachtime
                            break
                    last_allocation_time = eachtime
                    if eachtime >= period_start:
                        last_allocation = min(quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                    else:
                        last_allocation = min(last_quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))

                # 如果未计算到结束时间，则根据剩余迭代次数和 GPU 吞吐量估算结束时间
                if estimated_end_time is None:
                    new_allocations[last_allocation_time] = last_allocation
                    new_future_free_gpus[last_allocation_time] -= new_allocations[last_allocation_time]
                    assert new_future_free_gpus[last_allocation_time] >= 0
                    real_time_left = math.ceil(iter_left / float(
                        THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                        new_allocations[last_allocation_time])]))
                    time_left = utils.align_to_time_slot(0, real_time_left, FLAGS.scheduling_slot)
                    estimated_real_end_time = last_allocation_time + real_time_left
                    estimated_end_time = last_allocation_time + time_left
                new_allocations = OrderedDict(sorted(new_allocations.items(), key=lambda t: t[0]))

                # 更新最后一个事件的 GPU 数量
                last_event_gpu = CLUSTER.num_gpu
                for each_event_time in future_free_gpus:
                    if each_event_time < event_time:
                        continue
                    if each_event_time < estimated_end_time:
                        last_event_gpu = future_free_gpus[each_event_time]
                
                # 如果结束时间不在未来 GPU 自由量中，则添加该时间点，并保证 GPU 数量大于等于 0       
                if estimated_end_time not in future_free_gpus:
                    new_future_free_gpus[estimated_end_time] = last_event_gpu
                    assert new_future_free_gpus[estimated_end_time] >= 0
                new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
                
                # 如果是模拟，则更新作业字典中的相关字段；否则，更新集群的未来 GPU 自由量和作业的结束时间
                if simulation:
                    job_dict['next_level_allocation'] = utils.merge_dict(new_allocations)
                    job_dict['marginal_gputime'] = (utils.get_allocation_time(
                        new_allocations, estimated_end_time) - utils.get_allocation_time(
                        job_dict['allocations'], job_dict['end_time'])) / (job_dict['next_level_gpu'] - job_dict['num_gpu'])
                    job_dict['next_level_future_gpus'] = utils.merge_dict(new_future_free_gpus)
                    job_dict['next_level_endtime'] = estimated_end_time
                    job_dict['next_level_realendtime'] = estimated_real_end_time
                else:
                    if assign_gpu:
                        job_dict['old_end_time'] = job_dict['end_time']
                    CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                    job_dict['end_time'] = estimated_end_time
                    job_dict['real_end_time'] = estimated_real_end_time
                    job_dict['allocations'] = utils.merge_dict(new_allocations)
            
                return estimated_real_end_time <= job_dict['ddl'] # 如果估计的完成时间小于等于截止时间，则返回 True

            period_end = period_start

        last_last_quota = last_quota # 更新上上次的 GPU 配额
        last_quota = quota # 更新上次的 GPU 配额
        quota = utils.get_next_level(job_dict, gpu_num=quota) # 获取下一级的 GPU 配额
        if quota == last_quota: # 如果当前配额和上一次的配额相同，跳出循环
            break

    #if job_dict['job_idx'] == 
    # allocate from period start
    last_allocation_time = None
    last_allocation = None
    for eachtime in new_future_free_gpus:
        if eachtime > aligned_ddl:
             break # 如果超过对齐的截止时间，则结束循环
        if eachtime < event_time:
            continue # 如果时间小于事件时间，则继续下一次循环
                    
        if last_allocation_time is None:
            last_allocation_time = eachtime
            # 获取最后一个级别的 GPU 数量
            if eachtime >= point:
                last_allocation = min(last_quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            else:
                last_allocation = min(last_last_quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            continue
        # 更新分配情况和未来 GPU 自由量
        new_allocations[last_allocation_time] = last_allocation
        new_future_free_gpus[last_allocation_time] -= new_allocations[last_allocation_time]
        assert new_future_free_gpus[last_allocation_time] >= 0

        if last_allocation == 0:
            last_allocation_time = eachtime
            #last_allocation = min(last_quota, new_future_free_gpus[eachtime])
            #last_allocation = min(last_quota, 
            #    get_last_level(job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            # 获取最后一个级别的 GPU 数量
            if eachtime >= point:
                last_allocation = min(last_quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            else:
                last_allocation = min(last_last_quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            continue

        duration = eachtime - last_allocation_time
        if eachtime == aligned_ddl:
            duration = job_dict['ddl'] - last_allocation_time
        # 计算迭代次数
        iterations = duration * float(
            THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
            last_allocation)])
        if iterations >= iter_left:
            print("job", job_dict['job_idx'])
        assert iterations < iter_left
        iter_left -= iterations
        
        last_allocation_time = eachtime
        last_allocation = min(last_quota, utils.get_last_level(
            job_dict, gpu_num=new_future_free_gpus[eachtime]+1))

    assert iter_left > 0 # 确保迭代次数大于 0

    # allocate from ddl
    last_allocation_time = max(aligned_ddl, event_time) # 初始化上次分配时间为对齐的截止时间和事件时间的较大值
    for each_event_time in new_future_free_gpus: # 遍历新的未来可用 GPU 时间点
        if each_event_time <= aligned_ddl: # 如果时间点早于等于对齐的截止时间，则跳过
            continue
        # 获取当前时间点的最后一个级别的 GPU 数量，限制为作业的最大 GPU 数量
        quota = min(utils.get_last_level(
            job_dict, gpu_num=new_future_free_gpus[last_allocation_time]+1), job_dict['max_gpu'])
        if quota == 0: # 如果分配的 GPU 数量为 0
            new_allocations[last_allocation_time] = quota # 更新分配情况
            last_allocation_time = each_event_time # 更新上次分配时间为当前时间点
            continue
        # 计算当前时间点到下一个时间点的迭代次数
        new_allocations[last_allocation_time] = quota
        iterations = (each_event_time - last_allocation_time) * float(
            THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
            new_allocations[last_allocation_time])])
        
        if iterations >= iter_left: # 如果迭代次数超过剩余迭代次数
            # 估算实际结束时间和对齐结束时间
            estimated_real_end_time = math.ceil(last_allocation_time + iter_left / float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(quota)]))
            estimated_end_time = utils.align_to_time_slot(start_time, estimated_real_end_time, FLAGS.scheduling_slot)
            assert estimated_end_time <= each_event_time # 确保估算的结束时间早于等于当前时间点
            # 更新未来可用 GPU 时间点的 GPU 数量
            for each_event_time1 in new_future_free_gpus:
                if each_event_time1 < aligned_ddl:
                    continue
                if each_event_time1 < estimated_end_time:
                    if each_event_time1 in future_free_gpus:
                        last_event_gpu = future_free_gpus[each_event_time1]
                if each_event_time1 >= last_allocation_time and each_event_time1 < estimated_end_time:
                    new_future_free_gpus[each_event_time1] -= quota
                    assert new_future_free_gpus[each_event_time1] >= 0
            
            # 更新当前时间点的分配情况和 GPU 数量
            if last_allocation_time != estimated_end_time:
                new_future_free_gpus[last_allocation_time] = last_event_gpu - quota
            assert new_future_free_gpus[last_allocation_time] >= 0
            
            # 如果估算的结束时间不在未来可用 GPU 时间点中，则添加到字典中
            if estimated_end_time not in future_free_gpus:
                new_future_free_gpus[estimated_end_time] = last_event_gpu
                assert new_future_free_gpus[estimated_end_time] >= 0
            new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
            
            # 如果是模拟，则更新作业字典中的相关字段；否则更新集群的未来可用 GPU 和作业的结束时间
            if simulation:
                job_dict['next_level_allocation'] = utils.merge_dict(new_allocations)
                job_dict['marginal_gputime'] = (utils.get_allocation_time(
                    new_allocations, estimated_end_time) - utils.get_allocation_time(
                    job_dict['allocations'], job_dict['end_time'])) / (job_dict['next_level_gpu'] - job_dict['num_gpu'])
                job_dict['next_level_future_gpus'] = utils.merge_dict(new_future_free_gpus)
                job_dict['next_level_endtime'] = estimated_end_time
                job_dict['next_level_realendtime'] = estimated_real_end_time
            else:
                if assign_gpu:
                    job_dict['old_end_time'] = job_dict['end_time']
                CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                job_dict['end_time'] = estimated_end_time
                job_dict['real_end_time'] = estimated_real_end_time
                job_dict['allocations'] = utils.merge_dict(new_allocations)
            
            return estimated_real_end_time <= job_dict['ddl'] # 返回是否满足作业的截止时间要求
        else: # 如果迭代次数未超过剩余迭代次数
            iter_left -= iterations # 减去迭代次数
            for each_event_time1 in future_free_gpus: # 更新未来可用 GPU 时间点的 GPU 数量
                if each_event_time1 < last_allocation_time:
                    last_event_gpu = new_future_free_gpus[each_event_time1]
            for each_event_time1 in new_future_free_gpus:
                if each_event_time1 >= last_allocation_time and each_event_time1 < each_event_time:
                    new_future_free_gpus[each_event_time1] -= quota
                    assert new_future_free_gpus[each_event_time1] >= 0

        last_allocation_time = each_event_time # 更新上次分配时间为当前时间点
    
    assert iter_left > 0 # 断言剩余迭代次数大于 0
    available_gpu = CLUSTER.num_gpu # 可用 GPU 数量为集群中的 GPU 数量
    quota = min(CLUSTER.num_gpu, job_dict['max_gpu']) # 分配的 GPU 数量限制为集群中的 GPU 数量和作业的最大 GPU 数量的较小值
    if assign_gpu: # 如果要指定 GPU 数量
        if assigned_gpus > quota: # 如果已分配的 GPU 数量大于限制的 GPU 数量
            print("assigned_gpus", assigned_gpus, "quota", quota)  
            print("job", job_dict['job_idx'], "new_allocations", new_allocations) 
        assert assigned_gpus <= quota # 断言已分配的 GPU 数量不超过限制的 GPU 数量
        quota = assigned_gpus # 更新限制的 GPU 数量为已分配的 GPU 数量
        
    new_allocations[last_allocation_time] = quota # 更新分配情况
    new_allocations = OrderedDict(sorted(new_allocations.items(), key=lambda t: t[0])) # 按时间顺序排序分配情况
    
    # 估算实际结束时间和对齐结束时间
    estimated_real_end_time = math.ceil(last_allocation_time + iter_left / float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(quota)]))
    estimated_end_time = utils.align_to_time_slot(start_time, estimated_real_end_time, FLAGS.scheduling_slot)
    assert estimated_end_time > job_dict['ddl'] # 断言估算的结束时间晚于作业的截止时间
    return_value = False # 返回值初始化为 False
    new_future_free_gpus[last_allocation_time] = available_gpu - quota # 更新当前时间点的未来可用 GPU 数量
    assert new_future_free_gpus[last_allocation_time] >= 0 # 断言当前时间点的未来可用 GPU 数量大于等于 0
    new_future_free_gpus[estimated_end_time] = CLUSTER.num_gpu # 添加估算的结束时间到未来可用 GPU 中
    new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))  # 按时间顺序排序未来可用 GPU

    if simulation: # 如果是模拟
        # 更新作业字典中的相关字段
        job_dict['next_level_allocation'] = utils.merge_dict(new_allocations)
        gpu_increased = job_dict['next_level_gpu'] - job_dict['num_gpu']
        if gpu_increased == 0:
            job_dict['marginal_gputime'] = -sys.maxsize#sys.maxsize # GPU 增加为 0 时，设置为最小值
        else:
            job_dict['marginal_gputime'] = (utils.get_allocation_time(
                new_allocations, estimated_end_time) - utils.get_allocation_time(
                job_dict['allocations'], job_dict['end_time'])) / gpu_increased
        job_dict['next_level_future_gpus'] = utils.merge_dict(new_future_free_gpus)
        job_dict['next_level_endtime'] = estimated_end_time
        job_dict['next_level_realendtime'] = estimated_real_end_time
    else:
        if assign_gpu:
            job_dict['old_end_time'] = job_dict['end_time']
        CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus) # 更新集群中的未来可用 GPU 数量
        job_dict['end_time'] = estimated_end_time # 更新作业的结束时间
        job_dict['real_end_time'] = estimated_real_end_time # 更新作业的实际结束时间
        job_dict['allocations'] = utils.merge_dict(new_allocations) # 更新作业的分配情况

    return estimated_real_end_time <= job_dict['ddl'] # 返回是否满足作业的截止时间要求
       

def edf_sim_allocation(job_dict, start_time, simulation=True, future_free_gpus=None):
    event_time = start_time
    return_value = True
    global_batch_size = str(job_dict['batch_size'])
    if 'iter_left' not in job_dict:
        job_dict['iter_left'] = job_dict['iteration']
    #print("time", start_time, "job", job_dict['job_idx'], CLUSTER.future_free_gpus, CLUSTER.free_gpu)
    iter_left = job_dict['iter_left']

    new_allocations = {event_time:0}
    if future_free_gpus is None:
        future_free_gpus = CLUSTER.future_free_gpus
    new_future_free_gpus = copy.deepcopy(future_free_gpus)

    available_gpu = new_future_free_gpus[start_time]
    #available_gpu = get_available_gpu(start_time)
    base_quota = min(job_dict['max_gpu'], CLUSTER.num_gpu)

    for future_event_time in future_free_gpus:
        if future_event_time < event_time:
            available_gpu = future_free_gpus[future_event_time]
            del new_future_free_gpus[future_event_time]
            continue
        elif future_event_time == event_time:
            available_gpu = future_free_gpus[future_event_time]
            continue
        # the least number of GPU to meet DDL requirements
        duration = future_event_time - event_time    
        #print("job", job_dict['job_id'], "future_event_time", future_event_time, "base_quota", base_quota)
        #print("***FREE GPU", available_gpu, "at ", event_time)
        
        if available_gpu >= base_quota and base_quota > 0:
            new_allocation = base_quota
            new_allocations[event_time] = new_allocation
            estimated_end_time = math.ceil(event_time + iter_left / float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(base_quota)]))
            last_event_gpu = CLUSTER.num_gpu
            # if future_event_time > job_dict['end_time']
            if future_event_time >= estimated_end_time:
                for each_event_time in future_free_gpus:
                    if each_event_time < event_time:
                        continue
                    if each_event_time < estimated_end_time:
                        last_event_gpu = new_future_free_gpus[each_event_time]
                        new_future_free_gpus[each_event_time] -= base_quota
                        assert new_future_free_gpus[each_event_time] >= 0

                if event_time != estimated_end_time:
                    new_future_free_gpus[event_time] = available_gpu - base_quota
                assert new_future_free_gpus[event_time] >= 0
                if estimated_end_time not in future_free_gpus:
                    #new_future_free_gpus[estimated_end_time] = last_event_gpu
                    new_future_free_gpus[utils.align_to_time_slot(start_time, 
                        estimated_end_time, FLAGS.scheduling_slot)] = last_event_gpu
                    assert new_future_free_gpus[event_time] >= 0
                new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
                
                CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                job_dict['new_end_time'] = estimated_end_time
                job_dict['new_allocations'] = utils.merge_dict(new_allocations)
                    
                #print("event time:", event_time, "allocation:", job_dict['new_allocations'], "job",job_dict['job_id'], "end time", job_dict['new_end_time'])
                #print("future_free_gpus", new_future_free_gpus)
                if estimated_end_time > job_dict['ddl']:
                    #print("ERROR: Fail to meet deadline requirements!")
                    return_value = False
                return return_value
            else:
                #iterations = math.ceil(duration * float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                #    new_allocation)]))
                iterations = duration * float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                    new_allocation)])
                iter_left -= iterations
                if event_time in new_future_free_gpus:
                    new_future_free_gpus[event_time] -= new_allocation
                else:
                    new_future_free_gpus[event_time] = last_event_gpu - new_allocations[event_time]
                assert new_future_free_gpus[event_time] >= 0

        elif available_gpu >= job_dict["min_gpu"] and base_quota > 0:

            for throughput in THROUGHPUTS[job_dict['model']['name']][global_batch_size]:
                gpu_num = int(throughput)
                if gpu_num > available_gpu:
                    break
                #iterations = math.ceil(duration * float(
                #    THROUGHPUTS[job_dict['model']['name']][global_batch_size][throughput]))
                iterations = duration * float(
                    THROUGHPUTS[job_dict['model']['name']][global_batch_size][throughput])
                new_allocations[event_time] = gpu_num
            estimated_end_time = math.ceil(
                event_time + iter_left / float(
                    THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(new_allocations[event_time])]))
            last_event_gpu = CLUSTER.num_gpu
            # if future_event_time > job_dict['end_time']
            if future_event_time >= estimated_end_time:
                for each_event_time in future_free_gpus:
                    if each_event_time < event_time:
                        continue
                    if each_event_time < estimated_end_time:
                        last_event_gpu = new_future_free_gpus[each_event_time]
                        new_future_free_gpus[each_event_time] -= new_allocations[event_time]
                        assert new_future_free_gpus[each_event_time] >= 0

                new_future_free_gpus[event_time] = available_gpu - new_allocations[event_time]
                assert new_future_free_gpus[event_time] >= 0
                if estimated_end_time not in future_free_gpus:
                    #new_future_free_gpus[estimated_end_time] = last_event_gpu
                    new_future_free_gpus[utils.align_to_time_slot(start_time, 
                        estimated_end_time, FLAGS.scheduling_slot)] = last_event_gpu
                    assert new_future_free_gpus[event_time] >= 0
                new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
                
                CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                job_dict['new_end_time'] = estimated_end_time
                job_dict['new_allocations'] = utils.merge_dict(new_allocations)
                    
                #print("event time:", event_time, "allocation:", job_dict['new_allocations'], "job",job_dict['job_id'], "end time", job_dict['new_end_time'])
                #print("future_free_gpus", new_future_free_gpus)
                if estimated_end_time > job_dict['ddl']:
                    #print("ERROR: Fail to meet deadline requirements!")
                    return_value = False
                return return_value
            iter_left -= iterations
            new_future_free_gpus[event_time] -= new_allocations[event_time]
            assert new_future_free_gpus[event_time] >= 0
        elif available_gpu < job_dict["min_gpu"]:
            new_allocations[event_time] = 0
        available_gpu = new_future_free_gpus[future_event_time]
        event_time = future_event_time
        if event_time > job_dict['ddl']:
            #print("ERROR: Fail to meet deadline requirements!")
            return_value = False
    
    # another round
    available_gpu = CLUSTER.num_gpu

    assert base_quota <= available_gpu
    if base_quota > available_gpu:
        print("base_quota > available_gpu")
        return False

    # event time >= max(future event time)
    new_allocations[event_time] = base_quota
    estimated_end_time = math.ceil(event_time + iter_left / float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(base_quota)]))
    if estimated_end_time > job_dict['ddl']:
        #print("ERROR: Fail to meet deadline requirements!")
        return_value = False
    new_future_free_gpus[event_time] = available_gpu - base_quota
    assert new_future_free_gpus[event_time] >= 0
    #new_future_free_gpus[estimated_end_time] = CLUSTER.num_gpu
    new_future_free_gpus[utils.align_to_time_slot(start_time, 
        estimated_end_time, FLAGS.scheduling_slot)] = CLUSTER.num_gpu
    new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
    CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
    job_dict['new_end_time'] = estimated_end_time
    job_dict['new_allocations'] = utils.merge_dict(new_allocations)
        

    #print("event time:", event_time, "allocation:", job_dict['new_allocations'], "job",job_dict['job_id'],"end time", job_dict['new_end_time'])
    #print("future_free_gpus", new_future_free_gpus)
    return return_value


def get_marginal(job, allocatable_gpus, cur_time):
    if job['next_level_gpu'] - job['num_gpu'] > allocatable_gpus or job['next_level_gpu'] == job['num_gpu']:
        job['marginal_gputime'] = -sys.maxsize
        return
    # simulate to re-allocate the job
    last_allocated_gpu = 0
    tmp_future_free_gpus = copy.deepcopy(CLUSTER.future_free_gpus)
    if job['end_time'] not in tmp_future_free_gpus:
        for each_event_time in tmp_future_free_gpus:
            if each_event_time < job['end_time']:
                last_event_gpu = tmp_future_free_gpus[each_event_time]
            else:
                break
        tmp_future_free_gpus[job['end_time']] = last_event_gpu
        tmp_future_free_gpus = OrderedDict(sorted(tmp_future_free_gpus.items(), key=lambda t: t[0]))
    for eachallocation in job['allocations']:
        if eachallocation not in tmp_future_free_gpus:
            tmp_future_free_gpus[eachallocation] = get_available_gpu(
                eachallocation, future_free_gpus=tmp_future_free_gpus)
            tmp_future_free_gpus = OrderedDict(sorted(tmp_future_free_gpus.items(), key=lambda t: t[0]))
        for future_event_time in tmp_future_free_gpus:
            if future_event_time < eachallocation:
                continue
            if future_event_time >= job['end_time']:
                break
            delta = job['allocations'][eachallocation] - last_allocated_gpu
            tmp_future_free_gpus[future_event_time] += delta
        
        last_allocated_gpu = job['allocations'][eachallocation]
    for future_event_time in tmp_future_free_gpus:
        if tmp_future_free_gpus[future_event_time] > CLUSTER.num_gpu:
            print("tmp_future_free_gpus",tmp_future_free_gpus)
            print("CLUSTER.future_free_gpus", CLUSTER.future_free_gpus)
        assert tmp_future_free_gpus[future_event_time] <= CLUSTER.num_gpu
    tmp_future_free_gpus = utils.merge_dict(tmp_future_free_gpus, start_time=cur_time)
    ef_sim_allocation(job, cur_time, assign_gpu=True, assigned_gpus=job['next_level_gpu'], 
        simulation=True, future_free_gpus=tmp_future_free_gpus)
    if job['num_gpu'] == 0:
        job['marginal_gputime'] = sys.maxsize
    del tmp_future_free_gpus

def allocate_free_gpus(cur_time):
    total_gpu_used = 0
    for job in JOBS.runnable_jobs:
        #marginal_throughput(job, cur_time)
        job['next_level_gpu'] = utils.get_next_level(job)
        total_gpu_used += job['num_gpu']
        print("job", job['job_id'], job['num_gpu'])
    allocatable_gpus = CLUSTER.num_gpu - total_gpu_used
    assert allocatable_gpus >= 0
    print()
    print(allocatable_gpus, "gpus left")
    print("CLUSTER.future_free_gpus", CLUSTER.future_free_gpus)
    if CLUSTER.future_free_gpus[cur_time] != allocatable_gpus:
        for r_job in JOBS.runnable_jobs:
            print("^^^", r_job['job_idx'], r_job['allocations'])
    assert CLUSTER.future_free_gpus[cur_time] == allocatable_gpus
    # sort and find the ones to allocate more GPUS, change end time of these jobs
    if allocatable_gpus == 0:
        return
    for job in JOBS.runnable_jobs:
        if job['next_level_gpu'] - job['num_gpu'] > allocatable_gpus:
            print("not enough GPU for job", job['job_id'], job['next_level_gpu'] - job['num_gpu'])
            #continue
        if job['next_level_gpu'] == job['num_gpu']:
            print("Cannot scale up for job", job['job_id'])
            #continue
        get_marginal(job, allocatable_gpus, cur_time)
        
    #JOBS.running_jobs.sort(key = lambda e:e.__getitem__('marginal_throughput'), reverse=True)
    JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('marginal_gputime'), reverse=True)
    
    allocated_last_round = True
    while allocatable_gpus > 0 and allocated_last_round:
        allocated_last_round = False
        for eachjob in JOBS.runnable_jobs:
            if eachjob['next_level_gpu'] - eachjob['num_gpu'] > allocatable_gpus:
                print("not enough GPU for job", eachjob['job_id'], eachjob['next_level_gpu'] - eachjob['num_gpu'])
                continue
            if eachjob['next_level_gpu'] == eachjob['num_gpu']:
                print("Cannot scale up for job", eachjob['job_id'])
                continue
            # allocate
            print("\nscale job[", eachjob['job_idx'], "] up to", eachjob['next_level_gpu'])
            old_gpu_num = eachjob['num_gpu']

            del CLUSTER.future_free_gpus
            CLUSTER.future_free_gpus = utils.merge_dict(eachjob['next_level_future_gpus'])
            print("CLUSTER.future_free_gpus", CLUSTER.future_free_gpus)
            eachjob['allocations'] = eachjob['next_level_allocation']
            eachjob['num_gpu'] = eachjob['next_level_gpu']
            eachjob['old_end_time'] = eachjob['end_time']
            eachjob['end_time'] = eachjob['next_level_endtime']
            eachjob['real_end_time'] = eachjob['next_level_realendtime']
            if eachjob in JOBS.pending_jobs:
                JOBS.remove_from_pending(eachjob, cur_time)
            JOBS.change_job_end_event(eachjob)
            del eachjob['next_level_endtime'], eachjob['next_level_gpu'], eachjob['next_level_allocation']
            allocatable_gpus -= (eachjob['num_gpu'] - old_gpu_num)
            assert CLUSTER.future_free_gpus[cur_time] == allocatable_gpus
            allocated_last_round = True
            #marginal_throughput(eachjob, cur_time)
            eachjob['next_level_gpu'] = utils.get_next_level(eachjob)
            #if eachjob['next_level_gpu'] - eachjob['num_gpu'] > 0:
            #    get_marginal(eachjob, allocatable_gpus, cur_time)
            break
        for job in JOBS.runnable_jobs:
            """if job['next_level_gpu'] - job['num_gpu'] > allocatable_gpus:
                print("not enough GPU for job", job['job_id'], job['next_level_gpu'] - job['num_gpu'])
                continue"""
            if job['next_level_gpu'] - job['num_gpu'] > 0:
                get_marginal(job, allocatable_gpus, cur_time)

        #JOBS.running_jobs.sort(key = lambda e:e.__getitem__('marginal_throughput'), reverse=True)
        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('marginal_gputime'), reverse=True)

    print("after allocation, CLUSTER.future_free_gpus", CLUSTER.future_free_gpus)


def estimate_all_jobs(job_list, cur_time, record_old_end_time=True):
    del CLUSTER.future_free_gpus
    CLUSTER.future_free_gpus = {cur_time: CLUSTER.num_gpu}
    result = True
    job_list.sort(key = lambda e:e.__getitem__('ddl'))
    for job in job_list:
        if record_old_end_time:
            job['old_end_time'] = job['end_time']
        if 'node_set' in job:
            CLUSTER.release_job_res(job, end=False)
    for job in job_list:
        if not ef_sim_allocation(job, cur_time):
            result = False
    return result

def EDF_estimate_all_jobs(job_list, cur_time):
    del CLUSTER.future_free_gpus
    CLUSTER.future_free_gpus = {cur_time: CLUSTER.num_gpu}
    value = True
    job_list.sort(key = lambda e:e.__getitem__('ddl'))
    for eachjob in job_list:
        if not edf_sim_allocation(eachjob, cur_time):
            value = False
    return value


def get_available_gpu(event_time, future_free_gpus=None):
    if future_free_gpus is None:
        future_free_gpus = CLUSTER.future_free_gpus
    if event_time in future_free_gpus:
        return future_free_gpus[event_time]
    return_value = CLUSTER.num_gpu
    for each_event_time in future_free_gpus:
        if each_event_time <= event_time:
            return_value = future_free_gpus[each_event_time]
    return return_value


'''
Gandiva scheduler assumption
''' 
def gandiva_sim_jobs(gputime=False, solve_starvation=0):
    '''
    new jobs are added to the end of the ending queue
    but any fit job should be executed in fifo order
    '''
    global global_lock, this_round_begin_time, fast_forward_permission
    global schedule_count
    cur_time = JOBS.job_events[0]['time']
    node_release = False
    time_diff = 0
    last_reschedule_time = 0
    while (len(JOBS.job_events) + len(JOBS.pending_jobs) + len(JOBS.running_jobs))> 0:
        # if len(JOBS.job_events) == 0:
        #     break
        new_job_start = False
        CLUSTER.gandiva_node_set_adjust(cur_time, JOBS, LOG)
        print("%d-%d, %d, %d " % (cur_time, len(JOBS.job_events), len(JOBS.pending_jobs), len(JOBS.running_jobs)))
        #update job progress for end_jobs
        node_release = CLUSTER.time_slicing_execute(cur_time, JOBS, LOG, time_diff)
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0

        #get new start job
        event = utils.search_dict_list(JOBS.job_events, 'time', cur_time)
        event_time = cur_time
        if event != None:
            #for new-start jobs, try to start
            for s_job in event['start_jobs']:
                ret = try_get_job_res(s_job)
                if ret == False:
                    JOBS.move_to_pending(s_job)
                else:
                    s_job['start_time'] = cur_time
                    JOBS.running_jobs.append(s_job)
                    if 'best_effort' not in s_job or int(s_job['best_effort']) != 1:
                        JOBS.num_accepted_job += 1
                    utils.print_fn('----job[%d] starts' % s_job['job_idx'])

            #remove time_event
            JOBS.job_events.remove(event)

        if node_release: 
            for p_job in JOBS.pending_jobs:
                ret = try_get_job_res(p_job)
                if ret == True:
                    JOBS.remove_from_pending(p_job, cur_time)
                    p_job['start_time'] = cur_time
                    #JOBS.running_jobs.append(p_job)
                    utils.print_fn('----job[%d] starts from pending' % p_job['job_idx'])
                    new_job_start = True

        node_release = False
        
        # add node_set information to job_dict
        for node_set in CLUSTER.node_g:
            for each_set in CLUSTER.node_g[node_set]:
                concurrency = 0
                for each_job in each_set['jobs']:
                    concurrency = concurrency + 1
                    if concurrency <= each_set['capacity']:
                        each_job['node_set'] = each_set
                    else:
                        each_job['node_set'] = None
        #if event != None or new_job_start:
        LOG.scheduling_result(event_time)

        # change from 10 to time_slot
        if len(JOBS.job_events) <= 0:
            cur_time = cur_time + FLAGS.scheduling_slot
            time_diff = FLAGS.scheduling_slot
        else:
            if FLAGS.simulation:
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
                schedule_count += 1
                for r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart)
                    else:
                        r_job['overhead'] = 0
            else:
                global_lock.acquire()
                if not fast_forward_permission:
                    cur_time = last_reschedule_time
                    last_reschedule_time = event_time
                this_round_begin_time = math.ceil(time.time())
                global_lock.release()
                get_ef_input_no_overlap([], this_round_begin_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
                #start time of next job
                #next_s_time = JOBS.job_events[0]['time']
                next_s_time = cur_time + FLAGS.scheduling_slot
                for each_event in JOBS.job_events:
                    if len(each_event['start_jobs']) == 0:
                        continue
                    next_s_time = max(each_event['time'], next_s_time)
                    break

                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):
                    time.sleep(1)
                    cur_time += 1
                if not fast_forward_permission:
                    print("ATTENTION!!!, cur_time", cur_time)
                    update_overhead()

                for r_job in JOBS.running_jobs:
                    if r_job['job_idx'] not in job_stable:
                        if r_job['job_idx'] in this_round_running_jobs:
                            continue
                    r_job['old_end_time'] = r_job['end_time']
                    #if r_job['job_idx'] not in job_stable:
                    #    # not all jobs have finished scaling, but they have to be rescheduled
                    #    r_job['overhead'] = next_s_time - event_time
                    if r_job['job_idx'] in this_round_running_jobs:
                        r_job['end_time'] += r_job['overhead']

            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']
            
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                del JOBS.job_events[1]

            next_e_time = JOBS.job_events[0]['time']
            assert next_e_time >= cur_time
            time_diff = int(next_e_time - cur_time)
            cur_time = next_e_time
            LOG.checkpoint(event_time)

    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def one_queue_edf_sim_jobs():
    '''
    run jobs in edf order, without access control;
    jobs are sorted by ddl
    '''
    global this_round_begin_time, fast_forward_permission, global_lock
    global schedule_count
    idle_round = 0
    time_diff = 0
    cur_time = JOBS.job_events[0]['time']
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            if idle_round > 1:
                utils.print_fn("This cluster is not large enough to run the job")
                print(JOBS.pending_jobs)
                break
            idle_round += 1
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        print(cur_time)
        JOBS.run_all_jobs(time_diff, cur_time)
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0
        new_job_flag = False

        if len(JOBS.job_events) > 0:
            event = JOBS.job_events[0]
            event_time = event['time']
            # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
            #for ending jobs, release gpu
            has_ejob = False
            for e_job in event['end_jobs']:
                if 'node_set' not in e_job:
                    print(e_job)
                #job completes
                CLUSTER.release_job_res(e_job)
                if e_job['end_time'] > e_job['ddl']:
                    utils.print_fn('----job[%d]\'s DDL request is not satisfied. Declined.' % e_job['job_idx'])
                    print(e_job['end_time'], "v.s.", e_job['ddl'])
                    JOBS.move_to_declined(e_job)
                    JOBS.num_accepted_job -= 1
                    #input()
                else:
                    print("ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                JOBS.remove_running(e_job)
                LOG.job_complete(e_job, event_time)
                has_ejob = True

            #for new-start jobs, try to start
            available_gpu = CLUSTER.check_free_gpu()
            CLUSTER.future_free_gpus = {cur_time:CLUSTER.num_gpu}
            if len(event['start_jobs']) > 0:
                new_job_flag = True
                EDF_estimate_all_jobs(JOBS.runnable_jobs + event['start_jobs'], event_time)
                for s_job in event['start_jobs']:
                    JOBS.move_to_pending(s_job) #add into pending list
                    available_gpu -= s_job['new_allocations'][event_time]
        
        if not new_job_flag:
            EDF_estimate_all_jobs(JOBS.runnable_jobs, event_time)
        for r_job in JOBS.running_jobs:
            if r_job['num_gpu'] > 0:
                CLUSTER.release_job_res(r_job, end=False)
        run_jobs = []
        for r_job in JOBS.runnable_jobs:
            if 'new_allocations' not in r_job:
                print(r_job)
            r_job['allocations'] = r_job['new_allocations']
            del r_job['new_allocations']
            r_job['num_gpu'] = r_job['allocations'][event_time]
                
            r_job['old_end_time'] = r_job['end_time']
            r_job['end_time'] = r_job['new_end_time']
            del r_job['new_end_time']
            if r_job in JOBS.running_jobs:
                if r_job['num_gpu'] > 0:
                    #ret = try_get_job_res(r_job)
                    #assert ret
                    run_jobs.append(r_job)
                JOBS.change_job_end_event(r_job)
            else:
                assert r_job in JOBS.pending_jobs
                if r_job['num_gpu'] > 0:
                    JOBS.get_network_load(r_job)
                    #ret = try_get_job_res(r_job)
                    #assert ret
                    run_jobs.append(r_job)
                    JOBS.remove_from_pending(r_job, event_time)       
                    JOBS.add_job_end_event(r_job)
                    utils.print_fn('----job[%d] starts from pending' % r_job['job_idx'])
        run_jobs.sort(key = lambda e:e['num_gpu'], reverse=True)
        for r_job in run_jobs:
            ret = try_get_job_res(r_job)
            assert ret


        LOG.scheduling_result(event_time)
        JOBS.job_events.pop(0)

        #remove time_event
        if len(JOBS.job_events) > 0:
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            
        if len(JOBS.job_events) <= 0:
            time_diff = FLAGS.scheduling_slot
        else:
            if FLAGS.simulation:
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
                schedule_count += 1
                if restart:
                    LOG.cache = list()
                for r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        r_job['overhead'] = 0
                        #r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache'])
                        r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] += r_job['overhead']
                        JOBS.change_job_end_event(r_job)
                    else:
                        r_job['overhead'] = 0
            else:
                global_lock.acquire()
                this_round_begin_time = math.ceil(time.time())
                global_lock.release()
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
                #start time of next job
                next_s_time = cur_time + FLAGS.scheduling_slot
                for each_event in JOBS.job_events:
                    if len(each_event['start_jobs']) == 0:
                        continue
                    next_s_time = max(each_event['time'], next_s_time)
                    break
                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):
                    time.sleep(1)
                    cur_time += 1
                if not fast_forward_permission:
                    print("ATTENTION!!!, cur_time", cur_time)
                    update_overhead()

                for r_job in JOBS.running_jobs:
                    if r_job['job_idx'] not in job_stable:
                        if r_job['job_idx'] in this_round_running_jobs:
                            continue
                    r_job['old_end_time'] = r_job['end_time']
                    #if r_job['job_idx'] not in job_stable:
                    #    # not all jobs have finished scaling, but they have to be rescheduled
                    #    r_job['overhead'] = next_s_time - event['time']
                    r_job['end_time'] += r_job['overhead']
                    JOBS.change_job_end_event(r_job)

            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                JOBS.job_events[0]['end_jobs'].extend(
                    JOBS.job_events[1]['end_jobs'])
                del JOBS.job_events[1]

            next_e_time = JOBS.job_events[0]['time']
            time_diff = int(next_e_time - cur_time)
            cur_time = next_e_time
            LOG.checkpoint(event_time)
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def one_queue_edf_sim_jobs_access_control():
    '''
    run jobs in edf order, with access control;
    jobs are sorted by ddl
    '''
    global this_round_begin_time, fast_forward_permission, global_lock
    global schedule_count
    idle_round = 0
    time_diff = 0
    cur_time = JOBS.job_events[0]['time']
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            if idle_round > 1:
                utils.print_fn("This cluster is not large enough to run the job")
                print(JOBS.pending_jobs)
                break
            idle_round += 1
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        print(cur_time)
        JOBS.run_all_jobs(time_diff, cur_time)
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0
        for r_job in JOBS.runnable_jobs:
            r_job['old_end_time'] = r_job['end_time']
            if 'node_set' in r_job:
                CLUSTER.release_job_res(r_job, end=False)
            #else:
            #    assert r_job['num_gpu'] == 0
        if CLUSTER.check_free_gpu() != CLUSTER.num_gpu:
            for eachjob in JOBS.runnable_jobs:
                print(eachjob['job_idx'], eachjob['num_gpu'])
            print(CLUSTER.check_free_gpu(), CLUSTER.num_gpu)
        assert CLUSTER.check_free_gpu() == CLUSTER.num_gpu
        

        if len(JOBS.job_events) > 0:
            event = JOBS.job_events[0]
            event_time = event['time']
            # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
            #for ending jobs, release gpu
            has_ejob = False
            for e_job in event['end_jobs']:
                #job completes
                print("job", e_job['job_idx'], "ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                #CLUSTER.release_job_res(e_job)
                JOBS.remove_running(e_job)
                LOG.job_complete(e_job, event_time)
                #assert cur_time <= e_job['ddl']
                if e_job['end_time'] > e_job['ddl']:
                    JOBS.move_to_declined(e_job)
                    JOBS.num_accepted_job -= 1
                has_ejob = True

            #for new-start jobs, try to start
            available_gpu = CLUSTER.check_free_gpu()
            #CLUSTER.future_free_gpus = {cur_time:CLUSTER.num_gpu}
            new_job_flag = False
            if len(event['start_jobs']) > 1:
                print(event['start_jobs'])
                #input()
            for s_job in event['start_jobs']:
                if not EDF_estimate_all_jobs(JOBS.runnable_jobs + [s_job], event_time):
                    utils.print_fn('----job[%d]\'s DDL request cannot be satisfied. Declined.' % s_job['job_idx'])
                    JOBS.move_to_declined(s_job, remove_from_pending=False)
                    #input()
                else:
                    JOBS.move_to_pending(s_job) #add into pending list
                    new_job_flag = True
                    available_gpu -= s_job['new_allocations'][event_time]
                    assert s_job in JOBS.pending_jobs
                    for r_job in JOBS.running_jobs:

                        if 'new_allocations' not in r_job:
                            print(r_job)
                        r_job['allocations'] = r_job['new_allocations']
                        del r_job['new_allocations']
                        r_job['num_gpu'] = r_job['allocations'][event_time]
                
                        r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] = r_job['new_end_time']
                        del r_job['new_end_time']
                        assert r_job['end_time'] <= r_job['ddl']
                        if r_job['job_idx'] == 9:
                            print("!!!from", r_job['old_end_time'], "to", r_job['end_time'])
                            #input()
                        JOBS.change_job_end_event(r_job)
                    new_start_job = list()
                    for p_job in JOBS.pending_jobs:
                        if 'new_allocations' not in p_job:
                            print(p_job)
                        p_job['allocations'] = p_job['new_allocations']
                        del p_job['new_allocations']
                        p_job['num_gpu'] = p_job['allocations'][event_time]
                
                        p_job['old_end_time'] = p_job['end_time']
                        p_job['end_time'] = p_job['new_end_time']
                        if p_job['end_time'] > p_job['ddl']:
                            print(p_job)
                        assert p_job['end_time'] <= p_job['ddl']
                        del p_job['new_end_time']
                        if p_job['num_gpu'] > 0:
                            new_start_job.append(p_job)  
                    for ns_job in new_start_job:
                        JOBS.get_network_load(ns_job)
                        utils.print_fn('----job[%d] starts from pending' % ns_job['job_idx'])
                        JOBS.remove_from_pending(ns_job, event_time)       
                        JOBS.add_job_end_event(ns_job)
        
        if new_job_flag:
            for r_job in JOBS.runnable_jobs:
                if r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        ret = try_get_job_res(r_job)
                        assert ret
                else:
                    assert r_job in JOBS.pending_jobs
                    if r_job['num_gpu'] > 0:
                        ret = try_get_job_res(r_job)
                        assert ret
             
        else:
            EDF_estimate_all_jobs(JOBS.runnable_jobs, event_time)
            for r_job in JOBS.runnable_jobs:
                if cur_time > r_job['ddl']:
                    print(r_job)
                #assert cur_time <= r_job['ddl'] cannot be guaranteed because of overhead
                r_job['allocations'] = r_job['new_allocations']
                del r_job['new_allocations']
                r_job['num_gpu'] = r_job['allocations'][event_time]

                if r_job['num_gpu'] > 0:
                    ret = try_get_job_res(r_job)
                    assert ret
                    if r_job in JOBS.pending_jobs:
                        JOBS.get_network_load(r_job)
                        JOBS.remove_from_pending(r_job, event_time)       
                        JOBS.add_job_end_event(r_job)
                    
                        utils.print_fn('----job[%d] starts from pending' % r_job['job_idx'])
                        r_job['end_time'] = r_job['new_end_time']
                        del r_job['new_end_time']
                        #assert r_job['end_time'] <= r_job['ddl']
                        JOBS.change_job_end_event(r_job)
                    else:
                        #if len(event['start_jobs']) == 0:
                        #r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] = r_job['new_end_time']
                        del r_job['new_end_time']
                        #assert r_job['end_time'] <= r_job['ddl']
                        JOBS.change_job_end_event(r_job)

        LOG.scheduling_result(event_time)
        JOBS.job_events.pop(0)

        #remove time_event
        if len(JOBS.job_events) > 0:
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            
        if len(JOBS.job_events) <= 0:
            time_diff = 10
        else:
            if FLAGS.simulation:
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
                schedule_count += 1
                if restart:
                    LOG.cache = list()
                for r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache'])
                        r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] += r_job['overhead']
                        JOBS.change_job_end_event(r_job)
                    else:
                        r_job['overhead'] = 0
            else:
                global_lock.acquire()
                this_round_begin_time = math.ceil(time.time())
                global_lock.release()
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
                #start time of next job
                next_s_time = cur_time + FLAGS.scheduling_slot
                for each_event in JOBS.job_events:
                    if len(each_event['start_jobs']) == 0:
                        continue
                    next_s_time = max(each_event['time'], next_s_time)
                    break
                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):
                    time.sleep(1)
                    cur_time += 1
                if not fast_forward_permission:
                    print("ATTENTION!!!, cur_time", cur_time)
                    update_overhead()

                for r_job in JOBS.running_jobs:
                    if r_job['job_idx'] not in job_stable:
                        if r_job['job_idx'] in this_round_running_jobs:
                            continue
                    r_job['old_end_time'] = r_job['end_time']
                    #if r_job['job_idx'] not in job_stable:
                    #    # not all jobs have finished scaling, but they have to be rescheduled
                    #    r_job['overhead'] = next_s_time - event['time']
                    r_job['end_time'] += r_job['overhead']
                    if r_job['job_idx'] == 9:
                        print("from", r_job['old_end_time'], "to", r_job['end_time'])
                    JOBS.change_job_end_event(r_job)

            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                JOBS.job_events[0]['end_jobs'].extend(
                    JOBS.job_events[1]['end_jobs'])
                del JOBS.job_events[1]

            next_e_time = JOBS.job_events[0]['time']
            time_diff = int(next_e_time - cur_time)
            cur_time = next_e_time

            LOG.checkpoint(event_time)
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def get_ef_input_no_overlap(end_jobs, actual_time):
    global job_stable, fast_forward_permission, MASTER_PORT, commands
    global last_round_running_jobs, this_round_running_jobs, trainers_to_kill
    global global_lock, global_ready_lock
    global job_to_be_killed, schedule_count
    global last_round_gpu_allocations, gpu_allocations
    global_lock.acquire()
    # if there is need to reschedule
    return_flag = True
    for job in JOBS.running_jobs:
        if job['num_gpu'] == 0 or job['node_set'] is None:
            if job['job_idx'] in last_round_running_jobs:
                return_flag = False
                break
            continue
        if job['num_gpu'] < CLUSTER.num_gpu_p_node:
            if job['job_idx'] in last_round_running_jobs:
                if 'num_gpu' in last_round_running_jobs[job['job_idx']] and last_round_running_jobs[job['job_idx']]['num_gpu'] == job['num_gpu']:
                    # do not need to update in this round
                    continue
                else:
                    return_flag = False 
                    break
            else:
                return_flag = False 
                break
        else:
            if job['job_idx'] in last_round_running_jobs:
                # same number of nodes
                if len(last_round_running_jobs[job['job_idx']]['worker_id']) == len(job['node_set']['nodes']):
                    continue
                else:
                    return_flag = False 
                    break
            else:
                return_flag = False 
                break

    if return_flag:
        fast_forward_permission = True
        global_lock.release()
        return

    restart_trainers = (schedule_count % FLAGS.restart_threshold == 0) and schedule_count != 0
    # restart if jobs didn't restart successfully last round
    if not restart_trainers:
        restart_trainers = job_to_be_killed
    del this_round_running_jobs
    this_round_running_jobs = dict()
    if restart_trainers:
        scheduler_rpc_client.schedule('RE')
    else:
        for e_job in end_jobs:
            assert e_job['job_idx'] in last_round_running_jobs
            for each_worker in last_round_running_jobs[e_job['job_idx']]['worker_id']:
                command = ' '.join(["K", 
                    str(each_worker), str(e_job['job_idx'])])
                print(command)
                scheduler_rpc_client.schedule(command)
                job_to_be_killed = True
            del last_round_running_jobs[e_job['job_idx']]
    
    gpu_allocations = [[0 for gpu in range(CLUSTER.num_gpu_p_node)] for _ in range(CLUSTER.num_node)]
    del job_stable
    job_stable = dict()
    commands = []

    # new jobs
    for job in JOBS.running_jobs:
        if job['num_gpu'] == 0 or job['node_set'] is None:
            continue
        cmd = 'R'
        if job['num_gpu'] >= CLUSTER.num_gpu_p_node:
            for i in range(len(job['node_set']['nodes'])):
                compressed_gpu_list = 0
                for j in range(len(gpu_allocations[job['node_set']['nodes'][i].id])):
                    gpu_allocations[job['node_set']['nodes'][i].id][j] = 1
                    compressed_gpu_list += (1 << j)
                MASTER_PORT += 1
                command = ' '.join([cmd, str(job['node_set']['nodes'][i].id), job['model_name'], 
                    str(job['batch_size']), str(job['job_idx']), str(min(job['num_gpu'], CLUSTER.num_gpu_p_node)), str(len(job['node_set']['nodes'])), 
                    str(i), '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), str(int(job['iter_left'])), str(actual_time)])
                print(command)
                if job['job_idx'] not in this_round_running_jobs:
                    this_round_running_jobs[job['job_idx']] = {'worker_id':[]}
                this_round_running_jobs[job['job_idx']]['worker_id'].append(str(job['node_set']['nodes'][i].id))
                if job['job_idx'] not in job_stable:
                    job_stable[job['job_idx']] = 0
                fast_forward_permission = False
                #scheduler_rpc_client.schedule(command)
                commands.append(command)
        else:
            node_id = job['node_set']['nodes'][0].id
            if job['job_idx'] in this_round_running_jobs:
                continue

            allocated_gpu = 0
            compressed_gpu_list = 0
            for i in range(len(gpu_allocations[node_id])):
                if gpu_allocations[node_id][i] == 1:
                    continue
                allocated_gpu += 1
                gpu_allocations[node_id][i] = 1
                compressed_gpu_list += (1 << i)
                if allocated_gpu == job['num_gpu']:
                    break
            MASTER_PORT += 1
            command = ' '.join([cmd, str(node_id), job['model_name'], str(job['batch_size']), str(job['job_idx']), 
                str(min(job['num_gpu'], CLUSTER.num_gpu_p_node)), str(len(job['node_set']['nodes'])), '0', '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), 
                str(int(job['iter_left'])), str(actual_time)])
            print(command)
            tmp_dict = dict()
            tmp_dict['worker_id'] = [str(node_id)]
            tmp_dict['num_gpu'] = job['num_gpu']
            tmp_dict['compressed_gpu_list'] = compressed_gpu_list
            this_round_running_jobs[job['job_idx']] = tmp_dict
            if job['job_idx'] not in job_stable:
                job_stable[job['job_idx']] = 0
            fast_forward_permission = False
            #scheduler_rpc_client.schedule(command)
            commands.append(command)
    #TODO: let master stop old jobs for on-the-fly elastic trainers
    
    global_ready_lock.acquire()
    if len(this_round_running_jobs) > 0:
        trainers_to_kill = {}
    for job in this_round_running_jobs:
        trainers_to_kill[job] = []
        if 'num_gpu' in this_round_running_jobs[job]:
            for each_gpu in utils.fetch_GPU_list_to_int(this_round_running_jobs[job]['compressed_gpu_list']):
                if last_round_gpu_allocations[int(this_round_running_jobs[job]['worker_id'][0])][each_gpu] == 0:
                    continue
                trainers_to_kill[job].append(utils.get_global_rank(
                    this_round_running_jobs[job]['worker_id'][0],
                    each_gpu, CLUSTER.num_gpu_p_node))
        else:   
            for each_worker in this_round_running_jobs[job]['worker_id']:
                for each_gpu in range(CLUSTER.num_gpu_p_node):
                    if last_round_gpu_allocations[int(each_worker)][each_gpu] == 0:
                        continue
                    trainers_to_kill[job].append(utils.get_global_rank(
                    each_worker, each_gpu, CLUSTER.num_gpu_p_node))
    
    print("$$$ in no overlap, trainers to kill", trainers_to_kill)
    print("$$$ last_round_running_jobs", last_round_running_jobs)
    print("$$$ this_round_running_jobs", this_round_running_jobs)

    if not restart_trainers:
        for job in last_round_running_jobs:
            for each_worker in last_round_running_jobs[job]['worker_id']:
                command = 'K ' + str(each_worker) + ' ' + str(job)
                scheduler_rpc_client.schedule(command)
                job_to_be_killed = True
    else:
        job_to_be_killed = False
    
    if not job_to_be_killed:
        # run all commands
        for command in commands:
            scheduler_rpc_client.schedule(command)
        scheduler_rpc_client.schedule('F')
        scheduler_rpc_client.schedule('T')
        last_round_gpu_allocations = gpu_allocations
    fast_forward_permission = (len(commands) == 0)

    global_ready_lock.release()
    del last_round_running_jobs
    last_round_running_jobs = this_round_running_jobs
    #last_round_gpu_allocations = gpu_allocations
    schedule_count += 1
    global_lock.release()


def update_overhead():
    global this_round_running_jobs, fast_forward_permission, global_lock
    global_lock.acquire()
    if fast_forward_permission:
        global_lock.release()
        return
    
    for each_job in JOBS.running_jobs:
        if each_job['num_gpu'] == 0:
            continue
        if each_job['job_idx'] not in job_stable:
            if each_job['job_idx'] in this_round_running_jobs:
                each_job['overhead'] = 0
                continue
            each_job['overhead'] = FLAGS.scheduling_slot
        if job_stable[each_job['job_idx']] == 0:
            each_job['overhead'] = FLAGS.scheduling_slot
    global_lock.release()


def ef_sim_jobs_access_control():
    '''
    run jobs with elasticity with access control;
    new jobs are added to the end of the pending queue;
    unsatisfiable jobs are declined.
    '''
    global this_round_begin_time, fast_forward_permission, global_lock  # 全局变量声明
    global schedule_count  # 全局变量声明
    cur_time = JOBS.job_events[0]['time']  # 获取当前时间
    node_release = False  # 初始化节点释放标志为假
    time_diff = 0  # 初始化时间差为0
    while (len(JOBS.job_events) + len(JOBS.pending_jobs) + len(JOBS.running_jobs)) > 0:  # 当还有待处理的作业事件、挂起的作业和运行中的作业时循环执行
        # 输出接受的作业数量和拒绝的作业数量
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        # 输出当前时间、待处理作业事件数量、挂起作业数量和运行中作业数量
        print("%d-%d, %d, %d " % (cur_time, len(JOBS.job_events), len(JOBS.pending_jobs), len(JOBS.running_jobs)))
        # 更新所有作业的进度
        JOBS.run_all_jobs(time_diff, cur_time)
        # 重置运行中的作业的额外开销为0
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0
        
        # 获取新的开始作业
        event = utils.search_dict_list(JOBS.job_events, 'time', cur_time)  # 查找指定时间的作业事件
        if event != None:  # 如果找到了作业事件
            JOBS.job_events.remove(event)  # 从作业事件列表中移除该事件
            if len(JOBS.pending_jobs) == 0 and len(JOBS.running_jobs) == 0:  # 如果没有挂起的作业和运行中的作业
                if len(event['end_jobs']) == 0 and len(event['start_jobs']) == 0:  # 如果作业事件既没有结束作业也没有开始作业
                    continue  # 继续下一次循环
            # 处理结束作业
            for e_job in event['end_jobs']:
                if 'node_set' in e_job:  # 如果结束作业指定了节点集合
                    # 从可迁移作业中移除
                    JOBS.remove_migratable(e_job)
                    # 作业完成
                    JOBS.remove_running(e_job)
                    CLUSTER.release_job_res(e_job)
                    LOG.job_complete(e_job, cur_time)
                    print("ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                    if e_job['real_end_time'] > e_job['ddl']:  # 如果实际结束时间超过了截止时间
                        print(e_job)
                        JOBS.move_to_declined(e_job)  # 将作业移动到被拒绝的列表中
                        JOBS.num_accepted_job -= 1

            for r_job in JOBS.runnable_jobs:
                r_job['old_end_time'] = r_job['end_time']
                r_job['old_allocations'] = r_job['allocations']

            new_job = False  # 初始化新作业标志为假
            if len(event['start_jobs']) > 0:  # 如果有开始作业
                CLUSTER.old_future_free_gpu = copy.deepcopy(CLUSTER.future_free_gpus)  # 备份未来自由GPU
                # 如果所有作业都无法估算
                if not estimate_all_jobs(JOBS.runnable_jobs + event['start_jobs'], cur_time):
                    # 处理开始作业
                    for s_job in event['start_jobs']:
                        if 'best_effort' in s_job and int(s_job['best_effort']) == 1:  # 如果是最佳尝试作业
                            JOBS.get_network_load(s_job)
                            JOBS.move_to_pending(s_job)  # 将作业移到挂起列表中
                        else:
                            utils.print_fn('----job[%d]\'s DDL request cannot be satisfied. Declined.' % s_job['job_idx'])
                            JOBS.move_to_declined(s_job)  # 将作业移到被拒绝的列表中
                    if not new_job:
                        CLUSTER.future_free_gpus = CLUSTER.old_future_free_gpu
                        del CLUSTER.old_future_free_gpu
                else:
                    new_job = True  # 设置新作业标志为真
                    for s_job in event['start_jobs']:
                        JOBS.get_network_load(s_job)
                        JOBS.move_to_pending(s_job)

            if not new_job:  # 如果没有新作业
                estimate_all_jobs(JOBS.runnable_jobs, cur_time, record_old_end_time=(len(event['start_jobs']) == 0))
            del CLUSTER.future_free_gpus
            CLUSTER.future_free_gpus = {cur_time: CLUSTER.num_gpu}
            JOBS.runnable_jobs.sort(key=lambda e: e.__getitem__('ddl'))  # 按照截止时间排序可运行作业
            for r_job in JOBS.runnable_jobs:
                r_job['num_gpu'] = r_job['allocations'][cur_time]  # 获取当前时间点的GPU数量
                JOBS.change_job_end_event(r_job)  # 更新作业的结束事件
                r_job['old_end_time'] = r_job['end_time']
                ef_sim_allocation(r_job, cur_time)  # 进行弹性作业分配
                if 'allocations' in r_job:
                    r_job['old_allocations'] = copy.deepcopy(r_job['allocations'])  # 备份作业的分配情况

            # 移除时间事件

        if CLUSTER.check_free_gpu() > 0:  # 如果有空闲的GPU
            # 对于挂起的作业，尝试启动
            new_start_list = list()  # 初始化新开始作业列表
            for p_job in JOBS.pending_jobs:  # 遍历挂起的作业
                pend_flag = False  # 初始化挂起标志为假
                allocation_time = cur_time + 1  # 初始化分配时间为当前时间加1
                for allocation_time in p_job['allocations']:  # 遍历作业的分配时间
                    if p_job['allocations'][allocation_time] == 0:  # 如果该时间点的分配为0
                        continue  # 继续下一个时间点
                    if allocation_time > cur_time:  # 如果分配时间大于当前时间
                        pend_flag = True  # 设置挂起标志为真
                    break
                p_job['num_gpu'] = p_job['allocations'][cur_time]  # 获取当前时间点的GPU数量
                CLUSTER.free_gpu = CLUSTER.check_free_gpu()  # 获取当前的空闲GPU数量
                if p_job['num_gpu'] <= CLUSTER.free_gpu and not pend_flag:  # 如果作业需要的GPU数量小于等于当前空闲GPU数量且未挂起
                    new_start_list.append(p_job)  # 将作业添加到新开始作业列表中
                    print("pending job", p_job['job_id'], p_job['allocations'])  # 输出挂起作业信息
            
            for ns_job in new_start_list:  # 遍历新开始作业列表
                JOBS.remove_from_pending(ns_job, cur_time)  # 从挂起列表中移除作业
                JOBS.add_job_end_event(ns_job)  # 添加作业结束事件
                ## 添加下一个安排事件
                # 如果作业可迁移，将其添加到可迁移作业列表中
                JOBS.add_migratable(ns_job)  
                utils.print_fn('----job[%d] starts from pending' % ns_job['job_idx'])  # 输出作业开始信息
            
            for r_job in JOBS.runnable_jobs:  # 遍历可运行作业
                print(r_job['job_idx'], r_job['allocations'], r_job['end_time'])  # 输出作业信息
            # 分配空闲的GPU
            allocate_free_gpus(cur_time)  
            # 更新状态
            # CLUSTER.status_update()

            first_scale_event_time = sys.maxsize  # 初始化首次扩展事件时间为最大值
            JOBS.runnable_jobs.sort(key=lambda e: e['num_gpu'], reverse=True)  # 按照GPU数量降序排序可运行作业
            for job in JOBS.runnable_jobs:  # 遍历可运行作业
                # 添加下一个分配的扩展事件
                job['allocations'] = OrderedDict(sorted(job['allocations'].items(), key=lambda t: t[0]))  # 对作业的分配按照时间排序
                for each_allocation_time in job['allocations']:  # 遍历作业的分配时间
                    if each_allocation_time > cur_time:  # 如果分配时间大于当前时间
                        # JOBS.add_job_scale_event(job, each_allocation_time)  # 添加作业的扩展事件
                        if each_allocation_time < first_scale_event_time:  # 如果该分配时间比首次扩展事件时间小
                            first_scale_event_time = each_allocation_time  # 更新首次扩展事件时间
                        break
                if job["num_gpu"] >= CLUSTER.num_gpu_p_node:  # 如果作业需要的GPU数量大于等于节点上的GPU数量
                    ret = try_get_job_res(job)  # 尝试获取作业资源
                    assert ret  # 断言资源获取成功
                    if not ret:  # 如果资源获取失败
                        print('ERROR when allocating for job[%d]' % job['job_idx'])  # 输出错误信息
            for node_group in reversed(list(CLUSTER.node_g.keys())):  # 遍历节点组
                if node_group >= CLUSTER.num_gpu_p_node:  # 如果节点组的GPU数量大于等于节点上的GPU数量
                    continue  # 继续下一次循环
                for job in JOBS.runnable_jobs:  # 遍历可运行作业
                    if job["num_gpu"] == node_group:  # 如果作业需要的GPU数量等于节点组的GPU数量
                        ret = try_get_job_res(job)  # 尝试获取作业资源
                        if not ret:  # 如果资源获取失败
                            print('ERROR when allocating for job[%d]' % job['job_idx'])  # 输出错误信息
                            print(CLUSTER.node_g)  # 输出节点组信息
                        assert ret  # 断言资源获取成功
            
        LOG.scheduling_result(cur_time)  # 记录调度结果
        JOBS.job_events.sort(key=lambda e: e.__getitem__('time'))  # 按照时间排序作业事件
        if len(JOBS.job_events) > 0:  # 如果还有作业事件
            if first_scale_event_time < JOBS.job_events[0]['time']:  # 如果首次扩展事件时间小于最早的作业事件时间
                JOBS.add_job_scale_event_new(first_scale_event_time)  # 添加新的作业扩展事件
                JOBS.job_events.sort(key=lambda e: e.__getitem__('time'))  # 按照时间排序作业事件
        else:  # 如果没有作业事件
            JOBS.add_job_scale_event_new(first_scale_event_time)  # 添加新的作业扩展事件
            JOBS.job_events.sort(key=lambda e: e.__getitem__('time'))  # 按照时间排序作业事件

        #input()

        if len(JOBS.job_events) <= 0:  # 如果没有作业事件
            cur_time = cur_time + 10  # 当前时间加10
            time_diff = 10  # 时间差为10
        else:  # 如果有作业事件
            if FLAGS.simulation:  # 如果是模拟模式
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0  # 检查是否需要重新启动
                schedule_count += 1  # 计数加1
                if restart:  # 如果需要重新启动
                    LOG.cache = list()  # 清空缓存
                for r_job in JOBS.runnable_jobs:  # 遍历可运行作业
                    if r_job['num_gpu'] > 0:  # 如果作业需要的GPU数量大于0
                        r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache'])  # 估算额外开销
                        # r_job['overhead'] = 0
                        r_job['old_end_time'] = r_job['end_time']  # 保存原始结束时间
                        r_job['end_time'] += r_job['overhead']  # 更新结束时间
                        JOBS.change_job_end_event(r_job)  # 更新作业的结束事件
                    else:  # 如果作业需要的GPU数量为0
                        r_job['overhead'] = 0  # 额外开销为0
            else:  # 如果不是模拟模式
                global_lock.acquire()  # 获取全局锁
                this_round_begin_time = math.ceil(time.time())  # 记录当前轮次的开始时间
                global_lock.release()  # 释放全局锁
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)  # 获取无重叠的弹性作业输入
                JOBS.job_events.sort(key=lambda e: e.__getitem__('time'))  # 按照时间排序作业事件
                # 下一个作业的开始时间
                next_s_time = cur_time + FLAGS.scheduling_slot  # 下一个作业的开始时间为当前时间加上调度槽大小
                for each_event in JOBS.job_events:  # 遍历作业事件
                    if len(each_event['start_jobs']) == 0:  # 如果作业事件中没有开始作业
                        continue  # 继续下一次循环
                    next_s_time = max(each_event['time'], next_s_time)  # 更新下一个作业的开始时间
                    break  # 跳出循环
                next_s_time = cur_time + FLAGS.scheduling_slot  # 下一个作业的开始时间为当前时间加上调度槽大小
                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):  # 如果不是快进模式或者快进权限未开启，并且当前时间小于下一个作业的开始时间
                    time.sleep(1)  # 休眠1秒
                    cur_time += 1  # 当前时间加1
                if not fast_forward_permission:  # 如果快进权限未开启
                    print("ATTENTION!!!, cur_time", cur_time)  # 输出注意信息
                    # 修改额外开销
                    update_overhead()  # 更新额外开销

                for r_job in JOBS.running_jobs:  # 遍历运行中的作业
                    if r_job['job_idx'] not in job_stable:  # 如果作业不稳定
                        if r_job['job_idx'] in this_round_running_jobs:  # 如果作业在本轮次运行中
                            continue  # 继续下一次循环
                    r_job['old_end_time'] = r_job['end_time']  # 保存原始结束时间
                    # 如果作业不稳定
                    #     # not all jobs have finished scaling, but they have to be rescheduled
                    #     r_job['overhead'] = next_s_time - event['time']
                    r_job['end_time'] += r_job['overhead']  # 更新结束时间
                    JOBS.change_job_end_event(r_job)  # 更新作业的结束事件

            JOBS.job_events.sort(key=lambda e: e.__getitem__('time'))  # 按照时间排序作业事件
            next_e_time = JOBS.job_events[0]['time']  # 获取下一个作业事件的时间
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:  # 如果下一个作业事件的时间与当前时间之差小于等于调度槽大小，并且作业事件列表的长度大于1
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot  # 更新作业事件的时间为当前时间加上调度槽大小
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:  # 如果下一个作业事件的时间与当前时间之差大于调度槽大小
                    break  # 跳出循环
                next_e_time = JOBS.job_events[1]['time']  # 更新下一个作业事件的时间
                JOBS.job_events[0]['start_jobs'].extend(JOBS.job_events[1]['start_jobs'])  # 合并开始作业列表
                JOBS.job_events[0]['end_jobs'].extend(JOBS.job_events[1]['end_jobs'])  # 合并结束作业列表
                del JOBS.job_events[1]  # 删除下一个作业事件
            next_e_time = JOBS.job_events[0]['time']  # 获取下一个作业事件的时间
            time_diff = int(next_e_time - event['time'])  # 计算时间差
            if time_diff < 0:  # 如果时间差小于0
                print("ATTENTION! time diff < 0", JOBS.job_events[0])  # 输出注意信息
            cur_time = next_e_time  # 当前时间更新为下一个作业事件的时间
    if not FLAGS.simulation:  # 如果不是模拟模式
        scheduler_rpc_client.schedule('S')  # 调度器进行调度




def ef_sim_jobs():
    '''
    run jobs with elasticity without access control;
    new jobs are added to the end of the pending queue;
    unsatisfiable jobs are declined.
    '''
    global this_round_begin_time, fast_forward_permission, global_lock
    global schedule_count
    cur_time = JOBS.job_events[0]['time']
    node_release = False
    time_diff = 0
    while (len(JOBS.job_events) + len(JOBS.pending_jobs) + len(JOBS.running_jobs))> 0:
        # if len(JOBS.job_events) == 0:
        #     break
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        print("%d-%d, %d, %d " % (cur_time, len(JOBS.job_events), len(JOBS.pending_jobs), len(JOBS.running_jobs)))
        #update job progress for end_jobs
        JOBS.run_all_jobs(time_diff, cur_time)
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0
        
        #get new start job
        event = utils.search_dict_list(JOBS.job_events, 'time', cur_time)
        if event != None:
            for e_job in event['end_jobs']:
                #remove from migratable jobs, if it's there
                JOBS.remove_migratable(e_job)
                #job completes
                CLUSTER.release_job_res(e_job)
                if e_job['end_time'] > e_job['ddl']:
                    utils.print_fn('----job[%d]\'s DDL request is not satisfied. Declined.' % e_job['job_idx'])
                    print(e_job['end_time'], "v.s.", e_job['ddl'])
                    JOBS.move_to_declined(e_job)
                    JOBS.num_accepted_job -= 1
                else:
                    print("ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                JOBS.remove_running(e_job)
                LOG.job_complete(e_job, cur_time)
                has_ejob = True
                #input()

            for r_job in JOBS.runnable_jobs:
                r_job['old_end_time'] = r_job['end_time']
                r_job['old_allocations'] = r_job['allocations']

            if len(event['start_jobs']) > 0:
                CLUSTER.old_future_free_gpu = copy.deepcopy(CLUSTER.future_free_gpus)
                for s_job in event['start_jobs']:
                    JOBS.move_to_pending(s_job)
            estimate_all_jobs(JOBS.runnable_jobs, cur_time)

            JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('ddl'))
            for r_job in JOBS.runnable_jobs:
                r_job['num_gpu'] = r_job['allocations'][cur_time]
                JOBS.change_job_end_event(r_job)
                r_job['old_end_time'] = r_job['end_time']
                if 'allocations' in r_job:
                    r_job['old_allocations'] = copy.deepcopy(r_job['allocations'])
            
            #remove time_event
            JOBS.job_events.remove(event)


        if CLUSTER.check_free_gpu() > 0:
            #for pending jobs, try to start
            new_start_list = list()
            for p_job in JOBS.pending_jobs:
                pend_flag = False
                allocation_time = cur_time + 1
                for allocation_time in p_job['allocations']:
                    if p_job['allocations'][allocation_time] == 0:
                        continue
                    if allocation_time > cur_time:
                        pend_flag = True
                    break
                p_job['num_gpu'] = p_job['allocations'][cur_time]
                if pend_flag:
                    JOBS.add_job_scale_event(p_job, allocation_time)
                CLUSTER.free_gpu = CLUSTER.check_free_gpu()
                if p_job['num_gpu'] <= CLUSTER.free_gpu and not pend_flag:
                    new_start_list.append(p_job)
                    print("pending job", p_job['job_id'], p_job['allocations'])
            
            for ns_job in new_start_list:
                JOBS.get_network_load(ns_job)
                JOBS.remove_from_pending(ns_job, cur_time)
                JOBS.add_job_end_event(ns_job)
                ## add next arrangement event
                #if job is migratable, add into migratable job list
                JOBS.add_migratable(ns_job)
                # JOBS.read_job_info(p_job['job_idx'])
                utils.print_fn('----job[%d] starts from pending' % ns_job['job_idx'])
            
            for r_job in JOBS.runnable_jobs:
                print(r_job['job_idx'], r_job['allocations'], r_job['end_time'])
            # allocate free GPUs
            allocate_free_gpus(cur_time)
            for job in JOBS.running_jobs:
                # add scaling event for next allocation
                job['allocations'] = OrderedDict(sorted(job['allocations'].items(), key=lambda t: t[0]))
                for each_allocation_time in job['allocations']:
                    if each_allocation_time > cur_time:
                        JOBS.add_job_scale_event(job, each_allocation_time)
                        break
                if job["num_gpu"] >= CLUSTER.num_gpu_p_node:
                    ret = try_get_job_res(job)
                    assert ret
                    if not ret:
                        print('ERROR when allocating for job[%d]' % job['job_idx'])
            for node_group in reversed(list(CLUSTER.node_g.keys())):
                if node_group >= CLUSTER.num_gpu_p_node:
                    continue
                for job in JOBS.running_jobs:
                    if job["num_gpu"] == node_group:
                        ret = try_get_job_res(job)
                        if not ret:
                            print('ERROR when allocating for job[%d]' % job['job_idx'])
                            print(CLUSTER.node_g)
                        assert ret
            
            LOG.scheduling_result(cur_time)
        
        #input()

        if len(JOBS.job_events) <= 0:
            cur_time = cur_time + 10
            time_diff = 10
        else:
            if FLAGS.simulation:
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
                schedule_count += 1
                if restart:
                    LOG.cache = list()
                for r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache'])
                        r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] += r_job['overhead']
                        JOBS.change_job_end_event(r_job)
                    else:
                        r_job['overhead'] = 0
            else:
                global_lock.acquire()
                this_round_begin_time = math.ceil(time.time())
                global_lock.release()
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
                #start time of next job
                next_s_time = cur_time + FLAGS.scheduling_slot
                for each_event in JOBS.job_events:
                    if len(each_event['start_jobs']) == 0:
                        continue
                    next_s_time = max(each_event['time'], next_s_time)
                    break
                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):
                    time.sleep(1)
                    cur_time += 1
                if not fast_forward_permission:
                    update_overhead()

                for r_job in JOBS.running_jobs:
                    if r_job['job_idx'] not in job_stable:
                        if r_job['job_idx'] in this_round_running_jobs:
                            continue
                    r_job['old_end_time'] = r_job['end_time']
                    #if r_job['job_idx'] not in job_stable:
                    #    # not all jobs have finished scaling, but they have to be rescheduled
                    #    r_job['overhead'] = next_s_time - event['time']
                    r_job['end_time'] += r_job['overhead']
                    JOBS.change_job_end_event(r_job)
            
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                JOBS.job_events[0]['end_jobs'].extend(
                    JOBS.job_events[1]['end_jobs'])
                del JOBS.job_events[1]
            next_e_time = JOBS.job_events[0]['time']
            time_diff = int(next_e_time - event['time'])
            if time_diff < 0:
                print("ATTENTION! time diff < 0", JOBS.job_events[0])
            assert time_diff >= 0
            cur_time = next_e_time
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def one_queue_fifo_sim_jobs():
    '''
    run jobs in fifo order;
    new jobs are added to the end of the pending queue
    '''
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            utils.print_fn("This cluster is not large enough to run the job")
            break

        event = JOBS.job_events[0]
        event_time = event['time']
        # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        has_ejob = False
        for e_job in event['end_jobs']:
            #remove from migratable jobs, if it's there
            # JOBS.remote_migratable(e_job)

            #job completes
            CLUSTER.release_job_res(e_job)
            # CLUSTER.release_gpus(e_job)
            LOG.job_complete(e_job, event_time)
            has_ejob = True


        #for new-start jobs, try to start
        for s_job in event['start_jobs']:
            #add into pending list
            JOBS.move_to_pending(s_job)


        if CLUSTER.check_free_gpu() > 0:
            #for pending jobs, try to start
            new_start_list = list()
            for p_job in JOBS.pending_jobs:
                # ret = CLUSTER.alloc_gpus(p_job)
                ret = try_get_job_res(p_job)
                if ret == True:
                    ''' if remove_from_pending, then will miss the next p_job in the list '''
                    new_start_list.append(p_job)
                    #if job is migratable, add into migratable job list
                    # JOBS.add_migratable(p_job)
                    # JOBS.remove_from_pending(p_job, event_time)
                    # JOBS.add_job_end_event(p_job)
                    # utils.print_fn('----job[%d] starts from pending' % p_job['job_idx'])
                    # JOBS.read_job_info(p_job['job_idx'])
                else:
                    break
            for ns_job in new_start_list:
                JOBS.remove_from_pending(ns_job, event_time)
                JOBS.add_job_end_event(ns_job)
                utils.print_fn('----job[%d] starts from pending' % ns_job['job_idx'])


        #sort pending jobs based on the num_gpu
        #JOBS.pending_jobs.sort(key = lambda e:e.__getitem__('num_gpu'))

        #remove time_event
        JOBS.job_events.pop(0)
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        # JOBS.print_job_events()

        LOG.checkpoint(event_time)


def themis_sim_jobs():
    '''
    themis finish-time fairness
    '''
    global this_round_begin_time, global_lock, fast_forward_permission
    num_steps_remaining_prev_iteration, isolated_throughputs_prev_iteration = {}, {}
    cumulative_isolated_time = {} 
    cur_time = JOBS.job_events[0]['time']
    schedule_count = 0
    old_running_jobs = []
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs)) > 0:
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        running_jobs_list = []
        for r_job in JOBS.runnable_jobs:
            if 'node_set' in r_job:
                CLUSTER.release_job_res(r_job, end=False)
        #jobs.run_all_jobs
        #cur_time = JOBS.job_events[0]['time'] #todo: judge
        print(cur_time, len(JOBS.job_events), "events left,", len(JOBS.runnable_jobs), "runnable jobs left")
        #if len(JOBS.job_events) == 0:
        #    input()
        event = utils.search_dict_list(JOBS.job_events, 'time', cur_time)
        if event != None:
            event = JOBS.job_events[0]
            event_time = event['time']
            # end job
            for e_job in event['end_jobs']:
                if e_job['end_time'] > e_job['ddl']:
                    utils.print_fn('----job[%d]\'s DDL request is not satisfied. Declined.' % e_job['job_idx'])
                    print(e_job['end_time'], "v.s.", e_job['ddl'])
                    JOBS.move_to_declined(e_job)
                    #JOBS.num_accepted_job -= 1
                else:
                    print("ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                    if 'best_effort' not in e_job or int(e_job['best_effort']) != 1:
                        JOBS.num_accepted_job += 1
                JOBS.remove_running(e_job)
                LOG.job_complete(e_job, e_job['end_time'])

            # end job, release all resources CLUSTER.release_job_res(e_job)
            for s_job in event['start_jobs']:
                #add into pending list
                JOBS.move_to_pending(s_job)
        if len(JOBS.runnable_jobs) > 0:
            scale_factors_array = utils.scale_factors_array(JOBS.runnable_jobs)
            isolated_throughputs = utils.get_isolated_throughputs(JOBS.runnable_jobs)
            x = cp.Variable(len(JOBS.runnable_jobs)) 
            #avg_share = math.ceil(CLUSTER.num_gpu / len(JOBS.runnable_jobs))
            job_idx = 0
            expected_time_fractions = []
            for r_job in JOBS.runnable_jobs:
                assert r_job['iter_left'] > 0
                if r_job['job_idx'] not in cumulative_isolated_time:
                    cumulative_isolated_time[r_job['job_idx']] = 0
                if r_job['job_idx'] in num_steps_remaining_prev_iteration:
                    cumulative_isolated_time[r_job['job_idx']] += (
                        num_steps_remaining_prev_iteration[r_job['job_idx']] -
                        r_job['iter_left']) / \
                        isolated_throughputs_prev_iteration[r_job['job_idx']]
                throughput = THROUGHPUTS[r_job['model']['name']][str(
                    r_job['batch_size'])][str(r_job['num_gpu'])]
                allocation_throughput = throughput * x[job_idx]
                expected_time_isolated = cumulative_isolated_time[r_job['job_idx']] + \
                (r_job['iter_left'] / isolated_throughputs[job_idx])
                expected_time_allocation = (event_time - r_job['submit_time']) + \
                    (r_job['iter_left'] * cp.inv_pos(allocation_throughput))
                num_steps_remaining_prev_iteration[r_job['job_idx']] = r_job['iter_left']
                expected_time_fraction = expected_time_allocation / expected_time_isolated
                #print("expected_time_allocation, expected_time_isolated", expected_time_allocation, expected_time_isolated)
                expected_time_fractions.append(expected_time_fraction)
                isolated_throughputs_prev_iteration[r_job['job_idx']] = isolated_throughputs[job_idx]
                job_idx += 1

            if len(expected_time_fractions) == 1:
                objective = cp.Minimize(expected_time_fractions[0])
            else:
                objective = cp.Minimize(cp.maximum(*expected_time_fractions))

            # Make sure that the allocation can fit in the cluster.
            constraints = utils.get_base_constraints(x, scale_factors_array)
            cvxprob = cp.Problem(objective, constraints)
            result = cvxprob.solve(solver='ECOS')

            if cvxprob.status != "optimal":
                print('WARNING: Allocation returned by policy not optimal!')
                
                
            print(x.value)
            # reset time so far
            """worker_time_so_far = 0
            for r_job in JOBS.runnable_jobs:
                r_job['job_time_so_far'] = FLAGS.scheduling_slot / 2.0
                worker_time_so_far += r_job['job_time_so_far']"""

            # update priorities
            #fractions = {}
            for i, r_job in enumerate(JOBS.runnable_jobs):
                """if worker_time_so_far == 0.0:
                    fraction = 0.0
                else:
                    fraction = r_job['job_time_so_far'] / worker_time_so_far
                fractions[r_job['job_idx']] = fraction
                new_priority = x.value[i] * 1e9
                if fractions[r_job['job_idx']] > 0.0:
                    new_priority = x.value[i] / fractions[r_job['job_idx']]"""
                r_job['priority'] = x.value[i] * 1e9
                if 'rounds_received' not in r_job:
                    r_job['rounds_received'] = 0
                if r_job['rounds_received'] > 0:
                    r_job['priority'] = x.value[i] / r_job['rounds_received']
                r_job['x'] = x.value[i]

            JOBS.runnable_jobs.sort(key=lambda e:(e.__getitem__('priority'), e.__getitem__('x')), reverse=True)
            JOBS.running_jobs = list()
            free_gpus = CLUSTER.num_gpu
            for r_job in JOBS.runnable_jobs:
                if free_gpus <= 0:
                    break
                assert free_gpus > 0
                if r_job['num_gpu'] <= free_gpus:
                    JOBS.running_jobs.append(r_job)
                    free_gpus -= r_job['num_gpu']
            # allocate
            JOBS.running_jobs.sort(key=lambda e:(e.__getitem__('num_gpu')), reverse=True)
            for r_job in JOBS.running_jobs:
                ret = try_get_job_res(r_job)
                if not ret:
                    print('ERROR when allocating for job[%d]' % r_job['job_idx'])
                    print(CLUSTER.node_g)
                assert ret
                running_jobs_list.append(r_job['job_idx'])
        LOG.scheduling_result(cur_time)

        if len(JOBS.job_events) > 0 and cur_time == JOBS.job_events[0]['time']:
            #remove time_event
            JOBS.job_events.pop(0)
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        # JOBS.print_job_events()

        # run all jobs
        next_e_time = cur_time + FLAGS.scheduling_slot # lease time
        time_diff = FLAGS.scheduling_slot
        
        if len(JOBS.job_events) > 0:
            next_s_time = JOBS.job_events[0]['time']
            while int(next_s_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    next_s_time = JOBS.job_events[0]['time']
                    break
                next_s_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                JOBS.job_events[0]['end_jobs'].extend(
                    JOBS.job_events[1]['end_jobs'])
                del JOBS.job_events[1]
            if int(next_s_time - cur_time) < FLAGS.scheduling_slot:
                assert len(JOBS.job_events) == 1
                JOBS.job_events[0]['time'] = next_e_time
            
        end_jobs = []
        reschedule_flag = False
        for job in old_running_jobs:
            if job not in running_jobs_list:
                reschedule_flag = True
                break
        for job in running_jobs_list:
            if job not in old_running_jobs:
                reschedule_flag = True
                break
        if FLAGS.simulation:
            restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
            schedule_count += 1
            
        elif reschedule_flag:
            global_lock.acquire()
            this_round_begin_time = math.ceil(time.time())
            global_lock.release()
            if event is None:
                get_ef_input_no_overlap([], this_round_begin_time)
            else:
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)####
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_e_time):
                time.sleep(1)
                cur_time += 1
            if not fast_forward_permission:
                update_overhead()

            for r_job in JOBS.running_jobs:
                if 'node_set' not in r_job:
                    continue
                #if r_job['job_idx'] not in job_stable:
                #    # not all jobs have finished scaling, but they have to be rescheduled
                #    r_job['overhead'] = next_e_time - event['time']
        for r_job in JOBS.runnable_jobs:
            if 'node_set' not in r_job:
                continue
            if reschedule_flag == False:
                r_job['overhead'] = 0
            elif 'overhead' not in r_job:
                r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache']) # only for simulation
            if 'remaining_time' not in r_job:
                r_job['remaining_time'] = r_job['duration']
            end_time = cur_time + r_job['overhead'] + r_job['remaining_time']
            r_job['remaining_time'] -= (time_diff - r_job['overhead'])
            r_job['remaining_time'] = max(0, r_job['remaining_time'])
            r_job['iter_left'] -= (time_diff - r_job['overhead']) * float(THROUGHPUTS[r_job['model']['name']][str(
                r_job['batch_size'])][str(r_job['num_gpu'])])
            if r_job['iter_left'] <= 0:
                r_job['remaining_time'] = 0
                end_time = next_e_time
            if end_time <= next_e_time:
                # find all jobs that will end before next scheduling event
                end_jobs.append(r_job)
                r_job['end_time'] = end_time
        if len(JOBS.job_events) > 0:
            if JOBS.job_events[0]['time'] == next_e_time:
                JOBS.job_events[0]['end_jobs'].extend(end_jobs)
            elif len(end_jobs) > 0:
                JOBS.job_events.append({'time':next_e_time,'start_jobs':[], 'end_jobs':end_jobs})
        elif len(end_jobs) > 0:
            JOBS.job_events.append({'time':next_e_time,'start_jobs':[], 'end_jobs':end_jobs})
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))

        cur_time = next_e_time
        old_running_jobs = running_jobs_list

        LOG.checkpoint(event_time)
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')

def get_tiresias_input(end_jobs, actual_time):
    global job_stable, fast_forward_permission, MASTER_PORT, commands
    global last_round_running_jobs, this_round_running_jobs, trainers_to_kill
    global global_lock, global_ready_lock
    global job_to_be_killed, schedule_count
    global last_round_gpu_allocations, gpu_allocations
    global_lock.acquire()
    # if there is need to reschedule
    return_flag = True
    for job in JOBS.runnable_jobs:
        if job['status'] != 'RUNNING':
            continue
        if job['num_gpu'] == 0 or job['node_set'] is None:
            if job['job_idx'] in last_round_running_jobs:
                return_flag = False
                break
            continue
        if job['num_gpu'] < CLUSTER.num_gpu_p_node:
            if job['job_idx'] in last_round_running_jobs:
                if 'num_gpu' in last_round_running_jobs[job['job_idx']] and last_round_running_jobs[job['job_idx']]['num_gpu'] == job['num_gpu']:
                    # do not need to update in this round
                    continue
                else:
                    return_flag = False 
                    break
            else:
                return_flag = False 
                break
        else:
            if job['job_idx'] in last_round_running_jobs:
                # same number of nodes
                if len(last_round_running_jobs[job['job_idx']]['worker_id']) == len(job['node_set']['nodes']):
                    continue
                else:
                    return_flag = False 
                    break
            else:
                return_flag = False 
                break

    if return_flag:
        fast_forward_permission = True
        global_lock.release()
        return

    restart_trainers = (schedule_count % FLAGS.restart_threshold == 0) and schedule_count != 0
    # restart if jobs didn't restart successfully last round
    if not restart_trainers:
        restart_trainers = job_to_be_killed
    del this_round_running_jobs
    this_round_running_jobs = dict()
    if restart_trainers:
        scheduler_rpc_client.schedule('RE')
    else:
        for e_job in end_jobs:
            assert e_job['job_idx'] in last_round_running_jobs
            for each_worker in last_round_running_jobs[e_job['job_idx']]['worker_id']:
                command = ' '.join(["K", 
                    str(each_worker), str(e_job['job_idx'])])
                print(command)
                scheduler_rpc_client.schedule(command)
                job_to_be_killed = True
            del last_round_running_jobs[e_job['job_idx']]
    
    gpu_allocations = [[0 for gpu in range(CLUSTER.num_gpu_p_node)] for _ in range(CLUSTER.num_node)]
    del job_stable
    job_stable = dict()
    commands = []

    # new jobs
    for job in JOBS.runnable_jobs:
        if job['status'] != 'RUNNING':
            continue
        if job['num_gpu'] == 0 or job['node_set'] is None:
            continue
        cmd = 'R'
        if job['num_gpu'] >= CLUSTER.num_gpu_p_node:
            for i in range(len(job['node_set']['nodes'])):
                compressed_gpu_list = 0
                for j in range(len(gpu_allocations[job['node_set']['nodes'][i].id])):
                    gpu_allocations[job['node_set']['nodes'][i].id][j] = 1
                    compressed_gpu_list += (1 << j)
                MASTER_PORT += 1
                command = ' '.join([cmd, str(job['node_set']['nodes'][i].id), job['model_name'], 
                    str(job['batch_size']), str(job['job_idx']), str(min(job['num_gpu'], CLUSTER.num_gpu_p_node)), str(len(job['node_set']['nodes'])), 
                    str(i), '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), str(int(job['iter_left'])), str(actual_time)])
                print(command)
                if job['job_idx'] not in this_round_running_jobs:
                    this_round_running_jobs[job['job_idx']] = {'worker_id':[]}
                this_round_running_jobs[job['job_idx']]['worker_id'].append(str(job['node_set']['nodes'][i].id))
                if job['job_idx'] not in job_stable:
                    job_stable[job['job_idx']] = 0
                fast_forward_permission = False
                #scheduler_rpc_client.schedule(command)
                commands.append(command)
        else:
            node_id = job['node_set']['nodes'][0].id
            if job['job_idx'] in this_round_running_jobs:
                continue

            allocated_gpu = 0
            compressed_gpu_list = 0
            for i in range(len(gpu_allocations[node_id])):
                if gpu_allocations[node_id][i] == 1:
                    continue
                allocated_gpu += 1
                gpu_allocations[node_id][i] = 1
                compressed_gpu_list += (1 << i)
                if allocated_gpu == job['num_gpu']:
                    break
            MASTER_PORT += 1
            command = ' '.join([cmd, str(node_id), job['model_name'], str(job['batch_size']), str(job['job_idx']), 
                str(min(job['num_gpu'], CLUSTER.num_gpu_p_node)), str(len(job['node_set']['nodes'])), '0', '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), 
                str(int(job['iter_left'])), str(actual_time)])
            print(command)
            tmp_dict = dict()
            tmp_dict['worker_id'] = [str(node_id)]
            tmp_dict['num_gpu'] = job['num_gpu']
            tmp_dict['compressed_gpu_list'] = compressed_gpu_list
            this_round_running_jobs[job['job_idx']] = tmp_dict
            if job['job_idx'] not in job_stable:
                job_stable[job['job_idx']] = 0
            fast_forward_permission = False
            #scheduler_rpc_client.schedule(command)
            commands.append(command)
    #TODO: let master stop old jobs for on-the-fly elastic trainers
    
    global_ready_lock.acquire()
    if len(this_round_running_jobs) > 0:
        trainers_to_kill = {}
    for job in this_round_running_jobs:
        trainers_to_kill[job] = []
        if 'num_gpu' in this_round_running_jobs[job]:
            for each_gpu in utils.fetch_GPU_list_to_int(this_round_running_jobs[job]['compressed_gpu_list']):
                if last_round_gpu_allocations[int(this_round_running_jobs[job]['worker_id'][0])][each_gpu] == 0:
                    continue
                trainers_to_kill[job].append(utils.get_global_rank(
                    this_round_running_jobs[job]['worker_id'][0],
                    each_gpu, CLUSTER.num_gpu_p_node))
        else:   
            for each_worker in this_round_running_jobs[job]['worker_id']:
                for each_gpu in range(CLUSTER.num_gpu_p_node):
                    if last_round_gpu_allocations[int(each_worker)][each_gpu] == 0:
                        continue
                    trainers_to_kill[job].append(utils.get_global_rank(
                    each_worker, each_gpu, CLUSTER.num_gpu_p_node))
    
    print("$$$ in no overlap, trainers to kill", trainers_to_kill)
    print("$$$ last_round_running_jobs", last_round_running_jobs)
    print("$$$ this_round_running_jobs", this_round_running_jobs)

    if not restart_trainers:
        for job in last_round_running_jobs:
            for each_worker in last_round_running_jobs[job]['worker_id']:
                command = 'K ' + str(each_worker) + ' ' + str(job)
                scheduler_rpc_client.schedule(command)
                job_to_be_killed = True
    else:
        job_to_be_killed = False
    
    if not job_to_be_killed:
        # run all commands
        for command in commands:
            scheduler_rpc_client.schedule(command)
        scheduler_rpc_client.schedule('F')
        scheduler_rpc_client.schedule('T')
        last_round_gpu_allocations = gpu_allocations
    fast_forward_permission = (len(commands) == 0)

    global_ready_lock.release()
    del last_round_running_jobs
    last_round_running_jobs = this_round_running_jobs
    #last_round_gpu_allocations = gpu_allocations
    schedule_count += 1
    global_lock.release()

def dlas_sim_jobs(gputime=False, solve_starvation=0):
    '''
    Job's executed time -- priority queue
    Q0:[0, 30min)
    Q1:[30min,1h)
    Q2:[1h, 2h)
    Q3:[2h, 00)
    in each queue, jobs are scheduled in fit-first with FIFO
    how to avoid starvation?
    TODO:  2. add move_back for avoiding starvation
    '''
    global this_round_begin_time, fast_forward_permission, global_lock, run_jobs
    global schedule_count
    end_events = list()
    next_job_jump = sys.maxsize
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            utils.print_fn("This cluster is not large enough to run the job")
            #print(JOBS.runnable_jobs)
            for each in JOBS.runnable_jobs:
                print(each['job_id'], each['num_gpu'], each['status'])
            break

        #decide which is the next event: start or end  ?
        start_event = None
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']

        end_event = None
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']
        
        event_time = sys.maxsize
        event = dict()
        event['time'] = sys.maxsize
        if end_time < start_time:
            event_time = end_time
            event = end_event
        elif end_time > start_time:        
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #decide if job_jump first or (start/end) first
        if event_time > next_job_jump:
            event_time = next_job_jump
            event = dict()

        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        print(event_time, event)
        cur_time = event_time

        # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                if e_job['end_time'] > e_job['ddl']:
                    utils.print_fn('----job[%d]\'s DDL request is not satisfied. Declined.' % e_job['job_idx'])
                    print(e_job['end_time'], "v.s.", e_job['ddl'])
                    JOBS.move_to_declined(e_job)
                    #input()
                else:
                    if 'best_effort' not in e_job or int(e_job['best_effort']) != 1:
                        JOBS.num_accepted_job += 1
                    print("ends at", event_time, e_job['end_time'], "ddl", e_job['ddl'])
                CLUSTER.release_job_res(e_job)
                LOG.job_complete(e_job, e_job['end_time'])
                # utils.print_fn('---- job[%d] is completed' % e_job['job_idx'])
                JOBS.runnable_jobs.remove(e_job)
                JOBS.queues[e_job['q_id']].remove(e_job)

        #for new jobs, append to runnable jobs with pending status
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job)
                s_job['q_id'] = 0 #any new start job should be in Q0
                JOBS.queues[0].append(s_job)
                utils.print_fn('---- job[%d] is added' % s_job['job_idx'])
            #pop start event
            JOBS.job_events.pop(0)

        #update executed_time
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if 'overhead' not in rjob:
                    rjob['overhead'] = 0
                tmp = max(int(event_time - rjob['last_check_time']) - rjob['overhead'], 0)
                rjob['total_executed_time'] = int(rjob['total_executed_time'] + tmp)
                rjob['executed_time'] = int(rjob['executed_time'] + tmp) # decide job priority queue
                rjob['last_check_time'] = event_time
                if rjob['overhead'] != 0:
                    rjob['overhead'] = 0

                #check demotion
                j_gt = 0
                if gputime:
                    j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                else:
                    j_gt = int(rjob['executed_time'])
                cur_qid = rjob['q_id']
                if cur_qid < int(JOBS.num_queue - 1): #not for the last queue 
                    if j_gt >= JOBS.queue_limit[cur_qid]:
                        rjob['q_id'] = int(cur_qid + 1)
                        JOBS.queues[rjob['q_id']].append(rjob)
                        JOBS.queues[cur_qid].remove(rjob)
                        print("job %d demote to Q%d" % (rjob['job_idx'], rjob['q_id']))

                if FLAGS.schedule == 'dlas-gpu-gittins': 
                    # rjob['rank'] = cal_r_gittins_index(JOBS.job_dist_data, j_gt)
                    rjob['rank'] = get_gittins_index(j_gt)

            elif 'PENDING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time']) 
                rjob['last_check_time'] = event_time
                rjob['pending_time'] = int(rjob['pending_time'] + tmp) #this is the total pending_time
                if rjob['executed_time'] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    rjob['last_pending_time'] = int(rjob['last_pending_time'] + tmp) #this is the total pending_time
                #Q0 job no need to push_back, and must be a runned 
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] > 0 and rjob['executed_time'] > 0:
                    if rjob['last_pending_time'] >= int(rjob['executed_time'] * solve_starvation):
                        rjob['executed_time'] = 0
                        rjob['last_pending_time'] = 0
                        JOBS.queues[0].append(rjob)
                        JOBS.queues[rjob['q_id']].remove(rjob)
                        rjob['q_id'] = 0
                        rjob['promote'] = int(rjob['promote'] + 1)

                if FLAGS.schedule == 'dlas-gpu-gittins': 
                    if gputime:
                        j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                    else:
                        j_gt = int(rjob['executed_time'])
                    # rjob['rank'] = cal_r_gittins_index(JOBS.job_dist_data, j_gt)
                    rjob['rank'] = get_gittins_index(j_gt)

            elif 'END' == rjob['status']: # won't happen
                JOBS.runnable_jobs.remove(rjob)
                # utils.print_fn('---- job[%d] completed' % rjob['job_idx'])
                pass

        #push job to their new queue
        # JOBS.update_priority_queues(gputime)

        ''' schedule jobs in each queue '''
        #empty_cluster resource
        #CLUSTER.empty_infra()
        CLUSTER.free_gpu = CLUSTER.num_gpu
        # for "count" placement
        run_jobs = list()
        preempt_jobs = list()

        # if FLAGS.schedule == 'dlas-gpu-gittins': 
        #     q = JOBS.queues[0]
        #     q.sort(key = lambda e:(e.__getitem__('rank'), e.__getitem__('r_submit_time')), reverse=True)

        if FLAGS.simulation:
            restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
            schedule_count += 1
        for queue in JOBS.queues:
            if FLAGS.schedule == 'dlas-gpu-gittins': 
                queue.sort(key = lambda e:(e.__getitem__('rank'), e.__getitem__('r_submit_time')), reverse=True)
            for job in queue:
                ## make sure that all jobs to run can be allocated
                #ret = CLUSTER.release_job_res(job, end=False)
                #assert ret
                if CLUSTER.free_gpu >= job['num_gpu']:
                    #should run
                    if job['status'] == 'PENDING':              
                        #not running
                        run_jobs.append(job)
                        if FLAGS.simulation:
                            job['overhead'] = utils.estimate_overhead(job['num_gpu'], restart)
                        
                    CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu'])
                else:
                    #should NOT run
                    if job['status'] == 'RUNNING':                   
                        #running
                        preempt_jobs.append(job)
                    continue
        for job in JOBS.runnable_jobs:
            if 'node_set' in job:
                # make sure that all jobs to run can be allocated
                ret = CLUSTER.release_job_res(job, end=False)
                assert ret
        
        for job in preempt_jobs:
            job['status'] = 'PENDING'
            # if job['q_id'] == 0:
            #     job['preempt'] = int(job['preempt'] + 1)
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            if job['start_time'] == sys.maxsize:
                job['start_time'] = event_time
            #JOBS.get_network_load(job)
            #ret = try_get_job_res(job)
            #assert ret
        JOBS.runnable_jobs.sort(key = lambda e:(e.__getitem__('num_gpu')), reverse=True)
        for job in JOBS.runnable_jobs:
            if job['status'] == 'RUNNING':
                JOBS.get_network_load(job)
                ret = try_get_job_res(job)
                if not ret:
                    print(CLUSTER.node_g)
                    print(job)
                assert ret

        #sort based on the job start time
        for queue in JOBS.queues:
            #job there are many students            
            pending_job = list()
            for job in queue: 
                # if sys.maxsize == job['start_time'] and job['status'] == 'PENDING':
                if job['status'] == 'PENDING':
                    pending_job.append(job)
                    # print(job['job_idx'])
            for job in pending_job: 
                queue.remove(job)
            queue.extend(pending_job)

        #update end events and sort, and get the most recent one
        del end_events[:]
        min_end_time = sys.maxsize
        tmp_end_event = dict()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                remaining_time = rjob['duration'] - rjob['total_executed_time']
                if FLAGS.simulation:
                    if restart:
                        rjob['overhead'] = utils.estimate_overhead(rjob['num_gpu'], restart)
                        remaining_time += rjob['overhead']
                end_time = int(event_time + remaining_time)
                end_event_time = max(event_time + FLAGS.scheduling_slot, end_time)
                if end_event_time < min_end_time:
                    tmp_end_event['time'] = end_event_time
                    tmp_end_event['end_jobs'] = list()
                    tmp_end_event['end_jobs'].append(rjob)
                    min_end_time = end_event_time
                    rjob['end_time'] = end_time
                elif min_end_time == end_event_time:
                    rjob['end_time'] = end_time
                    tmp_end_event['end_jobs'].append(rjob)
        if min_end_time < sys.maxsize:
            end_events.append(tmp_end_event)

        # what's the closest queue_jump (demotion, and promotion) among all the jobs
        next_job_jump = sys.maxsize
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                qid = rjob['q_id']
                if qid < int(JOBS.num_queue - 1):
                    if gputime:
                        jump_time = int(math.ceil((JOBS.queue_limit[qid] - rjob['executed_time'])/rjob['num_gpu']) + event_time)
                    else:
                        jump_time = int(JOBS.queue_limit[qid] - rjob['executed_time'] + event_time)
                    if jump_time < next_job_jump:
                        next_job_jump = jump_time

            elif 'PENDING' == rjob['status']: # when pending job will be push back to Q0
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] and rjob['executed_time'] > 0:
                    diff_time = int(rjob['executed_time'] * solve_starvation - rjob['last_pending_time'])
                    if diff_time > 0:
                        jump_time = int(diff_time + event_time)
                        if jump_time < next_job_jump:
                            next_job_jump = jump_time

        LOG.checkpoint(event_time)
        LOG.scheduling_tiresias_result(event_time)
        
        if not FLAGS.simulation:
            #decide which is the next event: start or end  ?
            start_time = sys.maxsize
            if len(JOBS.job_events) > 0:
                start_event = JOBS.job_events[0]
                start_time = start_event['time']
            end_event = None
            end_time = sys.maxsize
            if len(end_events) > 0:
                end_event = end_events[0]
                end_time = end_event['time']
        
            next_event_time = sys.maxsize
            if end_time < start_time:
                next_event_time = end_time
            elif end_time > start_time:        
                next_event_time = start_time
            elif end_time == start_time and end_time != sys.maxsize:
                next_event_time = start_time
            #decide if job_jump first or (start/end) first
            if event_time > next_job_jump:
                next_event_time = next_job_jump
            if next_event_time - cur_time < FLAGS.scheduling_slot:
                next_event_time = cur_time + FLAGS.scheduling_slot

            global_lock.acquire()
            this_round_begin_time = math.ceil(time.time())
            global_lock.release()
            if 'end_jobs' in event:
                get_tiresias_input(event['end_jobs'], this_round_begin_time)
            else:
                get_tiresias_input([], this_round_begin_time)
            while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_event_time):
                time.sleep(1)
                cur_time += 1
            if not fast_forward_permission:
                update_overhead()
                print("ATTENTION! not all jobs ready")

        # wait and record overhead.


        # if event time > start_time or end_time: modify start time and end time
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        if next_job_jump - event_time < FLAGS.scheduling_slot:
            next_job_jump = event_time + FLAGS.scheduling_slot
        if len(JOBS.job_events) > 0:
            next_e_time = JOBS.job_events[0]['time']
            while int(next_e_time - event_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = event_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - event_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                del JOBS.job_events[1]
            end_time = sys.maxsize
        assert len(end_events) == 1 or len(end_events) == 0
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']
            if len(end_events) == 1 and end_time < event_time + FLAGS.scheduling_slot:
                end_events[0]['time'] = event_time + FLAGS.scheduling_slot
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def get_gittins_index(a):
    job_info = JOBS.job_dist_data
    if a > job_info['data'][-2]:
        return 0
    idx = next(x[0] for x in enumerate(job_info['data']) if x[1] > a)
    return job_info['gittins'][idx]


def cal_r_gittins_index(job_data, a):
    '''
    a means attained-service to that job
    gittins_index = P/E
    r_gi = E/P
    '''
    ut_delta = JOBS.gittins_delta

    data = job_data['data']
    if a > (job_data['data'][-1] - 1):
        return 0.0
    else:
        idx = next(x[0] for x in enumerate(data) if x[1] > a)

    next_a = a + ut_delta
    if next_a > (job_data['data'][-1] - 1):
        idx_delta = job_data['num'] - 1
    else:
        idx_delta = next(x[0] for x in enumerate(data) if x[1] > next_a)
    # print(idx, idx_delta)

    p = round(((idx_delta - idx) * 1.0) / (job_data['num'] - idx), 5)

    e_sum = sum(data[idx : idx_delta]) + (ut_delta * (job_data['num'] - idx_delta))
    e = round(e_sum / (job_data['num'] - idx), 5)

    # rank of gittins index = 1/gi
    # r_gi = round(e / p, 4)
    r_gi = round(p * 1000000 / e, 4)

    # print(idx, idx_delta, p, e_sum, e, r_gi)
    return r_gi


def parse_job_dist():
    job_dist_file = os.path.join(os.getcwd(), 'yarn-gput1000.csv')
    fd = open(job_dist_file, 'r')
    reader = csv.DictReader(fd, delimiter = ',') 
    durations = list()
    for row in reader:
        durations.append(int(row['duration']))
    fd.close()
    total_len = len(durations)
    durations.sort()
    print("  %s samples are learned" % total_len)

    job_dict = dict()
    job_dict['num'] = total_len
    job_dict['data'] = durations

    gi = list()
    for v in job_dict['data']:
        gi.append(cal_r_gittins_index(job_dict, int(v-1)))

    # print(gi)
    job_dict['data'].append(sys.maxsize)
    gi.append(0.0)
    job_dict['gittins'] = gi

    return job_dict


def main():  # 定义主函数
    if FLAGS.schedule == 'multi-dlas-gpu':  # 如果调度策略是 'multi-dlas-gpu'
        if FLAGS.scheme != 'count':  # 如果方案不是 'count'
            utils.print_fn("In Main, multi-dlas-gpu without count")  # 打印消息 "In Main, multi-dlas-gpu without count"
            exit()  # 退出程序
    if FLAGS.gpu_type == 'A100':  # 如果 GPU 类型是 'A100'
        throughput_path = "./throughputs_A100/"  # 设置吞吐量文件路径为 "./throughputs_A100/"
    else:
        throughput_path = "./throughputs_T4/"  # 否则设置为 "./throughputs_T4/"
    for throughput_file in os.listdir(throughput_path):  # 遍历吞吐量文件路径下的文件列表
        profiles.parse_throughput_file(throughput_path + throughput_file)  # 解析各个文件中的吞吐量数据
    ''' Parse input'''  # 解析输入
    parse_job_file(FLAGS.trace_file)  # 解析作业文件
    parse_cluster_spec()  # 解析集群规格
    ''' prepare logging '''  # 准备日志记录
    LOG.init_log()  # 初始化日志记录
    # lp.placement(JOBS.job_list[0])
    ''' Prepare jobs'''  # 准备作业
    JOBS.prepare_job_start_events()  # 准备作业开始事件
    if FLAGS.schedule == 'edf':  # 如果调度策略是 'edf'
        #JOBS.job_dist_data = parse_job_dist()
        CLUSTER.init_gandiva_nodes()  # 初始化 Gandiva 节点
        one_queue_edf_sim_jobs()  # 执行 EDF 调度算法
    ################ 改这个
    elif FLAGS.schedule == 'ef-accessctrl':  # 如果调度策略是 'ef-accessctrl'
        CLUSTER.init_gandiva_nodes()  # 初始化 Gandiva 节点
        ef_sim_jobs_access_control()  # 执行带访问控制的 EF 调度算法
    elif FLAGS.schedule == 'ef':  # 如果调度策略是 'ef'
        CLUSTER.init_gandiva_nodes()  # 初始化 Gandiva 节点
        ef_sim_jobs()  # 执行 EF 调度算法
    elif FLAGS.schedule == 'edf-accessctrl':  # 如果调度策略是 'edf-accessctrl'
        CLUSTER.init_gandiva_nodes()  # 初始化 Gandiva 节点
        one_queue_edf_sim_jobs_access_control()  # 执行带访问控制的 EDF 调度算法
    elif FLAGS.schedule == 'fifo':  # 如果调度策略是 'fifo'
        one_queue_fifo_sim_jobs()  # 执行 FIFO 调度算法
    elif FLAGS.schedule == 'dlas':  # 如果调度策略是 'dlas'
        JOBS.job_dist_data = parse_job_dist()  # 解析作业分布数据
        dlas_sim_jobs()  # 执行 DLAS 调度算法
    elif FLAGS.schedule == 'dlas-gpu':  # 如           果调度策略是 'dlas-gpu'
        CLUSTER.init_gandiva_nodes()  # 初始化 Gandiva 节点
        dlas_sim_jobs(True)  # 执行带 GPU 的 DLAS 调度算法
    elif FLAGS.schedule == 'themis':  # 如果调度策略是 'themis'
        CLUSTER.init_gandiva_nodes()  # 初始化 Gandiva 节点
        themis_sim_jobs()  # 执行 Themis 调度算法
    elif FLAGS.schedule == 'gandiva':  # 如果调度策略是 'gandiva'
        CLUSTER.init_gandiva_nodes()  # 初始化 Gandiva 节点
        gandiva_sim_jobs(True, 1000)  # 执行 Gandiva 调度算法，传入参数 True 和 1000
    elif FLAGS.schedule == 'gpu-demands':  # 如果调度策略是 'gpu-demands'
        sim_gpu_demands()  # 执行 GPU 需求模拟
    else:
        one_queue_fifo_sim_jobs()  # 执行默认的 FIFO 调度算法
    print("accepted jobs:", JOBS.num_accepted_job)  # 打印接受的作业数量
    print("declined jobs:", JOBS.num_declined_job)  # 打印拒绝的作业数量
    # record time ratio, cluster size, trace_file, schedule, placement
    LOG.log_final_result(JOBS.num_accepted_job, JOBS.num_declined_job)  # 记录最终结果，包括接受的作业数量、拒绝的作业数量等信息

if __name__ == '__main__':  # 如果当前脚本作为主程序运行
    # print('Hello world %d' % 2)
    if not FLAGS.simulation:  # 如果不是模拟模式
        # RPC client to master
        scheduler_rpc_client = scheduler_client.SchedulerRpcClient('127.0.0.1', 6888)  # 创建调度器的 RPC 客户端对象
        # run master rpc server in the background
        scheduler_server_port = 6890  # 设置调度器服务器端口为 6890
        callbacks = {  # 设置回调函数字典
            'ReportStable' : report_stable_callback,  # 'ReportStable' 键对应 report_stable_callback 函数
            'ReportReady' : report_ready_callback,  # 'ReportReady' 键对应 report_ready_callback 函数
        }
        server_thread = threading.Thread(target=scheduler_server.serve,  # 创建后台线程，运行调度器的 RPC 服务器
            args=(scheduler_server_port, callbacks))
        server_thread.setDaemon(True)  # 将线程设置为守护线程
        server_thread.start()  # 启动线程
    main()  # 调用主函数开始执行脚本逻辑


# def main():

#     if FLAGS.schedule == 'multi-dlas-gpu': 
#         if FLAGS.scheme != 'count':
#             utils.print_fn("In Main, multi-dlas-gpu without count")
#             exit()
#     if FLAGS.gpu_type == 'A100':
#         throughput_path = "./throughputs_A100/"
#     else:
#         throughput_path = "./throughputs_T4/"
#     for throughput_file in os.listdir(throughput_path):
#         profiles.parse_throughput_file(throughput_path + throughput_file)
#     ''' Parse input'''
#     parse_job_file(FLAGS.trace_file)
#     parse_cluster_spec()

#     ''' prepare logging '''
#     LOG.init_log()

#     # lp.placement(JOBS.job_list[0])
#     ''' Prepare jobs'''
#     JOBS.prepare_job_start_events()

#     if FLAGS.schedule == 'edf':
#         #JOBS.job_dist_data = parse_job_dist()
#         CLUSTER.init_gandiva_nodes()
#         one_queue_edf_sim_jobs()
#     elif FLAGS.schedule == 'ef-accessctrl':
#         CLUSTER.init_gandiva_nodes()
#         ef_sim_jobs_access_control()
#     elif FLAGS.schedule == 'ef':
#         CLUSTER.init_gandiva_nodes()         
#         ef_sim_jobs()
#     elif FLAGS.schedule == 'edf-accessctrl':
#         CLUSTER.init_gandiva_nodes()
#         one_queue_edf_sim_jobs_access_control()
#     elif FLAGS.schedule == 'fifo':
#         one_queue_fifo_sim_jobs()
#     elif FLAGS.schedule == 'dlas':
#         JOBS.job_dist_data = parse_job_dist()
#         dlas_sim_jobs()
#     elif FLAGS.schedule == 'dlas-gpu':
#         CLUSTER.init_gandiva_nodes()
#         dlas_sim_jobs(True)
#     elif FLAGS.schedule == 'themis':
#         CLUSTER.init_gandiva_nodes()
#         themis_sim_jobs()
#     elif FLAGS.schedule == 'gandiva':
#         CLUSTER.init_gandiva_nodes()
#         gandiva_sim_jobs(True, 1000)
#     elif FLAGS.schedule == 'gpu-demands':
#         sim_gpu_demands()
#     else:
#         one_queue_fifo_sim_jobs()
#     print("accepted jobs:", JOBS.num_accepted_job)
#     print("declined jobs:", JOBS.num_declined_job)
#     # record time ratio, cluster size, trace_file, schedule, placement
#     LOG.log_final_result(JOBS.num_accepted_job, JOBS.num_declined_job)


# if __name__ == '__main__':
#     # print('Hello world %d' % 2)
#     if not FLAGS.simulation:
#         # RPC client to master
#         scheduler_rpc_client = scheduler_client.SchedulerRpcClient('127.0.0.1', 6888)
#         # run master rpc server in the background
#         scheduler_server_port = 6890
#         callbacks = {
#             'ReportStable' : report_stable_callback,
#             'ReportReady' : report_ready_callback,
#         }
#         server_thread = threading.Thread(target=scheduler_server.serve, 
#             args=(scheduler_server_port, callbacks))
#         server_thread.setDaemon(True)
#         server_thread.start()
#     main()
  
