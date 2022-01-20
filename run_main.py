import os
import argparse
import functools

import torch.distributed as dist

from DEE.utils import set_basic_log_config, strtobool
from DEE.DEE_task import DEETask, DEETaskSetting
from DEE.dee_helper import aggregate_task_eval_info, print_total_eval_info, print_single_vs_multi_performance

set_basic_log_config()


def chain_prod(num_list):
    return functools.reduce(lambda x, y: x * y, num_list)


def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task_name', type=str, default = 'SetPre4DEE_tf8_Mgpu_TrainNopair_turn_new',
                            help='Take Name')
    arg_parser.add_argument('--data_dir', type=str, default='./Data',
                            help='Data directory')
    arg_parser.add_argument('--exp_dir', type=str, default='./Exps',
                            help='Experiment directory')
    arg_parser.add_argument('--save_cpt_flag', type=strtobool, default=True,
                            help='Whether to save cpt for each epoch')
    arg_parser.add_argument('--eval_model_names', type=str, default='SetPre4DEE',
                            help="Models to be evaluated, seperated by ','")
    arg_parser.add_argument('--re_eval_flag', type=strtobool, default=False,
                            help='Whether to re-evaluate previous predictions')
    arg_parser.add_argument('--event_type_weight', type=list, default=[1, 0.2],
                            help='dict containing as key the names of the losses and as values their relative weight.')
    arg_parser.add_argument('--resume_latest_cpt', type=strtobool, default=True,
                            help = '# whether to resume latest checkpoints when training for fault tolerance')
    arg_parser.add_argument('--skip_train', type=strtobool, default=True,
                            help='Whether to skip training')
    arg_parser.add_argument('--num_set_decoder_layers', type=int, default=2)
    arg_parser.add_argument('--num_role_decoder_layers', type=int, default=4)
    arg_parser.add_argument('--num_generated_sets', type=int, default=5)
    arg_parser.add_argument('--use_pgd', type=strtobool, default=False,
                            help = 'whether use adversaral training')
    arg_parser.add_argument('--cost_weight', type=dict, default={'event_type': 1, 'role': 0.5},
                            help = 'cost weight for type and role')
    arg_parser.add_argument('--train_on_multi_events', type=strtobool, default=True,
                            help = 'whether only train only on multi-events datasets')
    arg_parser.add_argument('--train_on_single_event', type=strtobool, default=True,
                            help = 'whether only train only on single-event datasets')
    arg_parser.add_argument('--event_type_classes', type=int, default=2)
    arg_parser.add_argument('--train_on_multi_roles', type=strtobool, default=False)
    arg_parser.add_argument('--use_event_type_enc', type=strtobool, default=True)
    arg_parser.add_argument('--use_role_decoder', type=strtobool, default=True)
    arg_parser.add_argument('--use_sent_span_encoder', type=strtobool, default=False)
    arg_parser.add_argument('--start_epoch', type=int, default=0, help = 'start cpt model and save id')
    arg_parser.add_argument('--parallel_decorate', action="store_true", help='whether to use DataParallel or DistributedDataParallel')

    # add task setting arguments
    for key, val in DEETaskSetting.base_attr_default_pairs:
        if isinstance(val, bool):
            arg_parser.add_argument('--' + key, type=strtobool, default=val)
        else:
            arg_parser.add_argument('--'+ key, type=type(val), default=val)

    arg_info = arg_parser.parse_args(args=in_args)

    return arg_info


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '9'
    in_argv = parse_args()
    task_dir = os.path.join(in_argv.exp_dir, in_argv.task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir, exist_ok=True)

    in_argv.model_dir = os.path.join(task_dir, "Model")
    in_argv.output_dir = os.path.join(task_dir, "Output")

    # in_argv must contain 'data_dir', 'model_dir', 'output_dir'
    dee_setting = DEETaskSetting(
        **in_argv.__dict__
    )
    # eval_result_file_path = os.path.join(in_argv.output_dir, '{}_result.json'.format(in_argv.start_epoch))
    # default_dump_result_json(dee_setting, eval_result_file_path)
    # build task
    dee_task = DEETask(
        dee_setting,
        load_train=not in_argv.skip_train,
        parallel_decorate=in_argv.parallel_decorate
    )
    index2entity_label = dee_task.index2entity_label

    if not in_argv.skip_train:
        # dump hyper-parameter settings
        if dee_task.is_master_node():
            fn = '{}_{}.task_setting.json'.format(dee_setting.cpt_file_name,in_argv.start_epoch)
            dee_setting.dump_to(task_dir, file_name=fn)
            fn = '{}_result.json'.format(in_argv.start_epoch)
            dee_setting.dump_to(in_argv.output_dir, file_name=fn)
        dee_task.train(save_cpt_flag=in_argv.save_cpt_flag)
    else:
        dee_task.logging('Skip training')
        dee_task.resume_save_eval_at(epoch=dee_setting.start_epoch, resume_cpt_flag=True, save_cpt_flag=False, dee_setting=dee_setting)

    if in_argv.re_eval_flag:
        doc_type2data_span_type2model_str2epoch_res_list = dee_task.reevaluate_dee_prediction(dump_flag=True)
    else:
        doc_type2data_span_type2model_str2epoch_res_list = aggregate_task_eval_info(in_argv.output_dir, dump_flag=True)

    doc_type = "overall"
    data_type = 'test'
    span_type = 'pred_span'
    metric_type = 'micro'
    mstr_bepoch_list = print_total_eval_info(
        doc_type2data_span_type2model_str2epoch_res_list,
        metric_type=metric_type, span_type=span_type,
        model_strs=in_argv.eval_model_names.split(','),
        target_set=data_type
    )
    sm_results = print_single_vs_multi_performance(
        mstr_bepoch_list, in_argv.output_dir, dee_task.test_features, index2entity_label,
        metric_type=metric_type, data_type=data_type, span_type=span_type
    )

    # ensure every processes exit at the same time
    if dist.is_initialized():
        dist.barrier()
