#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heapq as hq
import numpy
import pandas as pd
import functools

import time
import copy
import sys
from itertools import chain
from sys import getsizeof

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import magellan as mg
from magellan.core.mtable import MTable

from collections import namedtuple
from collections import defaultdict
from operator import attrgetter

import token_based_jac_cython

def iterative_topK_debug_blocker(ltable, rtable, candidate_set, outdir, pred_list_size=100, field_corres_list=None):
    """
    Debug the blocker. The basic idea is trying to suggest the user a list of record pairs
    out of the candidate set with high (document) jaccard similarity. The object of similarity
    measurement (document) is generated based on a string concatenation method with field
    selection. Given the suggestion list, the user should go through the pairs and determine if
    there are, and how many true matches in it. And based on this information, the user can
    determine if further improvement on the blocking step is necessary (Ex. if there are many
    true matches in the list, the user may conclude that the blocking step is flawed, and should
    revise it to produce a better candidate set).

    Parameters
    ----------
    ltable, rtable : MTable
        Input MTables
    candidate_set : MTable
        The candidate set table after performing blocking on ltable and rtable
    pred_list_size : int
        The size of the output suggestion list
    field_corres_list : list (of tuples), defaults to None
        The list of field pairs from ltable and rtable. Each pair indicates a field correspondence
        between two tables. Since ltable and rtable can have different schemas, it' necessary to
        have this parameter to build the field correspondence to make sure the string concatenation
        algorithm runs correctly.
        Note each pair in the list should be represented as a tuple in the following format:
                            (some_ltable_field, some_rtable_field)

    Returns
    -------
    suggestion_table : MTable
        Contains a list of pair suggestions with high jaccard similarity. The output MTable contains
        the following fields:
            * _id
            * similarity (of the record pair)
            * ltable record key value
            * rtable record key value
            * field pairs from filtered corres_list (removing the numeric types)
                ltable_field_1
                rtable_field_1 (corresponding to ltable_field_1)
                ltable_field_2
                rtable_field_2 (corresponding to ltable_field_2)
                      .
                      .
                ltable_field_k
                rtable_field_k (corresponding to ltable_field_k)
    """

    # Basic checks.
    if len(ltable) == 0:
        raise StandardError('Error: ltable is empty!')
    if len(rtable) == 0:
        raise StandardError('Error: rtable is empty!')
    if pred_list_size <= 0:
        raise StandardError('The input parameter: \'pred_list_size\' is less than or equal to 0. Nothing needs to be done!')

    logging.info('\nPreparing for debugging blocker')

    # Check the user input field correst list (if exists) and get the raw version of
    # our internal correst list.
    check_input_field_correspondence_list(ltable, rtable, field_corres_list)
    corres_list = get_field_correspondence_list(ltable, rtable, field_corres_list)

    # Build the (col_name: col_index) dict to speed up locating a field in the schema.
    ltable_col_dict = build_col_name_index_dict(ltable)
    rtable_col_dict = build_col_name_index_dict(rtable)

    # Filter correspondence list to remove numeric types. We only consider string types
    # for document concatenation.
    filter_corres_list(ltable, rtable, ltable_col_dict, rtable_col_dict, corres_list)
    # logging.info('\nFiltered field correspondence list:\n' + str(corres_list))

    # Get field filtered new table.
    ltable_filtered, rtable_filtered = get_filtered_table(ltable, rtable, corres_list)

    # Select features.
    """TODO(hanli): currently we don't select the key fields even if they have the largest score.
    # This is because the values of the key field could be simply domain-specific serial numbers,
    # which might be meaningless or even harmful (when two tables use different key formats).
    # Modify it if this ituition is not proper."""
    feature_list = select_features(ltable_filtered, rtable_filtered)
    if len(feature_list) == 0:
        raise StandardError('\nError: the selected field list is empty, nothing could be done! ' +
                            'Please check if all table fields are numeric types.')
    # logging.info('\nSelected fields for concatenation:\n' + str([(ltable_filtered.columns[i],
    #  rtable_filtered.columns[i]) for i in feature_list]))
    print 'selected_fields:', ltable_filtered.columns[feature_list]

    # Build field_index_list for removing field
    field_index_list = range(len(feature_list))

    logging.info('\nTokenizing records')
    # Generate record lists
    lrecord_list, lrecord_id_to_index_map = get_tokenized_table_new(ltable_filtered, ltable.get_key(), feature_list)
    rrecord_list, rrecord_id_to_index_map = get_tokenized_table_new(rtable_filtered, rtable.get_key(), feature_list)

    logging.info('\nIndexing candidate sets')
    ltable_key = candidate_set.get_property('foreign_key_ltable')
    rtable_key = candidate_set.get_property('foreign_key_rtable')
    new_formatted_candidate_set = index_candidate_set(
         candidate_set, lrecord_id_to_index_map, rrecord_id_to_index_map, ltable_key, rtable_key)

    logging.info('\nBuilding token global order')
    # Build token global order
    order_dict = {}
    build_global_token_order(lrecord_list, order_dict)
    build_global_token_order(rrecord_list, order_dict)
    #print lrecord_list

    logging.info('\nSorting tables')
    # Sort each record by the token global order
    sort_record_tokens_by_global_order(lrecord_list, order_dict)
    sort_record_tokens_by_global_order(rrecord_list, order_dict)
    # print lrecord_list[1: 6]
    # print rrecord_list[1: 6]

    #removed_field = -1
    #remained_fields = list(field_index_list)
    #if removed_field in field_index_list:
    #    remained_fields.remove(removed_field)
    #topK_heap, compared_dict = run_iteration(lrecord_list, rrecord_list, {}, 0, None, pred_list_size, removed_field, remained_fields)
    #return

    basic_stat_list = []
    topK_heap, compared_dict, basic_stat = run_iteration(ltable_filtered, rtable_filtered, lrecord_list, rrecord_list,
                                             feature_list, {}, 0,
                                             new_formatted_candidate_set, pred_list_size, -1, field_index_list, outdir)
    basic_stat_list.append(basic_stat)
    topK_heap = sorted(topK_heap, key=lambda tup: tup[0], reverse=True)
    #for tuple in topK_heap:
    #    print tuple, list(ltable_filtered.ix[tuple[1]]), list(rtable_filtered.ix[tuple[2]])
        #print tuple, list(lrecord_list[tuple[1]]), list(rrecord_list[tuple[2]])
    #output_topK_heap(ltable_filtered, rtable_filtered, topK_heap, '../datasets/CS784/S_hanli/topK_results/topK_full.txt')

    for field_index in field_index_list:
        remained_fields = list(field_index_list)
        remained_fields.remove(field_index)
        topK_heap, new_compared_dict, basic_stat = run_iteration(ltable_filtered, rtable_filtered, lrecord_list, rrecord_list,
                      feature_list, compared_dict, 0,
                      new_formatted_candidate_set, pred_list_size, field_index, remained_fields, outdir)
        basic_stat_list.append(basic_stat)

    logging.info('\nOutputing basic stat')
    output_basic_stat(ltable_filtered, feature_list, basic_stat_list, outdir)
    logging.info('\nFinishing debugging blocking')

    return None


def run_iteration(ltable_filtered, rtable_filtered, lrecord_list, rrecord_list, feature_list,
                  compared_set, lower_bound, candidates, pred_list_size, removed_field, remained_fields,
                  outdir):
    logging.info(('\nRecaculating the updated length of records'))
    lrecord_length_list, lrecord_ave_length = calc_record_length(lrecord_list, remained_fields)
    rrecord_length_list, rrecord_ave_length = calc_record_length(rrecord_list, remained_fields)

    logging.info('\nGenerating prefix events')
    #Generate prefix events
    lprefix_events = []
    rprefix_events = []
    generate_prefix_events(lrecord_list, lrecord_length_list, lprefix_events, remained_fields, lower_bound)
    generate_prefix_events(rrecord_list, rrecord_length_list, rprefix_events, remained_fields, lower_bound)
    print removed_field, len(lprefix_events), len(rprefix_events)

    logging.info('\n>>>Performing sim join>>>')
    topK_heap = [(-1, -1, -1)]
    topK_heap, new_compared_dict, basic_stat = perform_sim_join(
                                 lrecord_list, rrecord_list, lrecord_length_list, rrecord_length_list,
                                 lprefix_events, rprefix_events, compared_set, candidates,
                                 topK_heap, pred_list_size, removed_field, remained_fields)
    logging.info('\n<<<Finishing sim join<<<')
    print topK_heap[0]
    topK_heap = sorted(topK_heap, key=lambda tup: tup[0], reverse=True)
    #for tuple in topK_heap:
    #    print tuple, list(ltable_filtered.ix[tuple[1]]), list(rtable_filtered.ix[tuple[2]])
    #    print tuple, list(lrecord_list[tuple[1]]), list(rrecord_list[tuple[2]])
    if removed_field < 0:
        output_topK_heap(ltable_filtered, rtable_filtered, feature_list, topK_heap, outdir + 'topK_full.txt')
    else:
        output_topK_heap(ltable_filtered, rtable_filtered, feature_list, topK_heap, outdir + 'topK_remove_' + str(removed_field) + '.txt')

    basic_stat.append(lrecord_ave_length)
    basic_stat.append(rrecord_ave_length)
    return topK_heap, new_compared_dict, basic_stat


def output_basic_stat(ltable_filtered, feature_list, basic_stat_list, outdir):
    f = open(outdir + 'basic_stat.txt', 'w')
    selected_fields = list(ltable_filtered.columns[feature_list])
    f.write(','.join(selected_fields) + '\n')
    cols = len(basic_stat_list)
    rows = len(basic_stat_list[0])
    for i in range(rows):
        for j in range(cols - 1):
            f.write(str(basic_stat_list[j][i]) + ',')
        f.write(str(basic_stat_list[cols - 1][i]) + '\n')
    f.close()

    # for tuple in basic_stat_list:
    #     length = len(tuple)
    #     for i in range(length - 1):
    #         f.write(str(tuple[i]) + ',')
    #     f.write(str(tuple[length - 1]) + '\n')
    #f.close()


def output_topK_heap(ltable_filtered, rtable_filtered, feature_list, topK_heap, output_name):
    out = open(output_name, 'w')
    count = 1
    for tuple in topK_heap:
        out.write('======Tuple ' + str(count) + "======\n")
        out.write(str(tuple[1]) + ' ' + str(tuple[2]) + ' ' + str(tuple[0]) + '\n')
        out.write(str(list(ltable_filtered.ix[tuple[1]][feature_list])) + "\n")
        out.write(str(list(rtable_filtered.ix[tuple[2]][feature_list])) + "\n\n")
        count += 1
    out.close()


def perform_sim_join(lrecord_list, rrecord_list, lrecord_length_list, rrecord_length_list, lprefix_events, rprefix_events,
                     compared_set, candidates, topK_heap, pred_list_size, removed_field, remained_fields):
    sim_join_start = time.clock()
    inverted_index = {}

    # The compared dict for this new iteration
    new_compared_dict = {}
    reused_compared_set = set([])

    lvisited_tokens_index = {}
    total_compared_pairs = 0
    reused_ignored_pairs = 0
    reused_total_pairs = 0
    new_calculated_pairs = 0
    #step_time = [0, 0]
    #jac_calc_time = [0, 0, 0]
    #update_inc_index_time = 0
    #gene_inc_index_time = 0

    while len(lprefix_events) > 0 and topK_heap[0][0] < lprefix_events[0][0] * -1:
    #while len(lprefix_events) > 0 and -100 < lprefix_events[0][0] * -1 :
        #print topK_heap[0], lprefix_events[0][0] * -1
        '''TODO(hanli): should consider rprefix_events size 0'''
        #gene_inc_index_start = time.clock()
        inc_inverted_index = {}
        while len(lprefix_events) > 0 and lprefix_events[0][0] >= rprefix_events[0][0]:
            r_pre_event = hq.heappop(rprefix_events)
            key = rrecord_list[r_pre_event[1]][r_pre_event[2]][0]
            if key not in inc_inverted_index:
                inc_inverted_index[key] = set()
            inc_inverted_index[key].add(r_pre_event[1])
            #if r_pre_event[0] < rprefix_events[0][0]:
            #    break
            if rprefix_events[0][0] > lprefix_events[0][0]:
                break
        #gene_inc_index_end = time.clock()
        #gene_inc_index_time += (gene_inc_index_end - gene_inc_index_start)
        #print inc_inverted_index

        for key in inc_inverted_index:
            if key in lvisited_tokens_index:
                lvisited_records = lvisited_tokens_index[key]
                inc_records = inc_inverted_index[key]
                for lindex in lvisited_records:
                    for rindex in inc_records:
                        #if lindex in compared_dict and rindex in compared_dict[lindex]:
                        #    continue
                        cur_record_pair = (lindex, rindex)

                        if cur_record_pair in candidates:
                            continue

                        if cur_record_pair in new_compared_dict:
                            continue

                        if cur_record_pair in reused_compared_set:
                            continue

                        total_compared_pairs += 1
                        if total_compared_pairs % 100000 == 0:
                            print total_compared_pairs, topK_heap[0], lprefix_events[0][0] * -1, len(lprefix_events), len(rprefix_events)
                            print get_total_size(new_compared_dict) / (1024 * 1024), get_total_size(inverted_index) / (1024 * 1024)

                        if cur_record_pair in compared_set:
                            '''The pair hasn't been compared, but compared in the upper level iteration'''
                            reused_total_pairs += 1

                            llen = lrecord_length_list[lindex]
                            rlen = rrecord_length_list[rindex]
                            old_result = compared_set[cur_record_pair]
                            # if llen + rlen - old_result[0] == 0:
                            #     print cur_record_pair
                            #     print key
                            #     print llen, rlen, old_result
                            #     print lrecord_list[lindex]
                            #     print rrecord_list[rindex]
                            #     exit(0)

                            sim_upper_bound = 1.0
                            denom = llen + rlen - old_result[0]
                            if denom > 0:
                                sim_upper_bound = old_result[0] * 1.0 / denom
                            if sim_upper_bound < topK_heap[0][0]:
                                reused_ignored_pairs += 1
                                reused_compared_set.add(cur_record_pair)
                                continue

                            old_count = old_result[0]
                            old_field_map = copy.deepcopy(old_result[1])
                            for old_field in old_field_map.keys():
                                if old_field == removed_field:
                                    for matched_field in old_field_map[old_field]:
                                        old_count -= old_field_map[old_field][matched_field]
                                    old_field_map.pop(old_field)
                                else:
                                    if removed_field in old_field_map[old_field]:
                                        old_count -= old_field_map[old_field].pop(removed_field)
                            new_compared_dict[cur_record_pair] = (old_count, old_field_map)
                            jac_sim = old_count * 1.0 / (llen + rlen - old_count)
                            if len(topK_heap) == pred_list_size:
                                hq.heappushpop(topK_heap, (jac_sim, lindex, rindex))
                            else:
                                hq.heappush(topK_heap, (jac_sim, lindex, rindex))
                        else:
                        #if True:
                            #cur_time = [0, 0, 0]
                            '''The pair is not compared at all. Run the complete compair procedure'''
                            new_calculated_pairs += 1

                            # cur_lrecord = lrecord_list[lindex]
                            # new_cur_lrecord = []
                            # for i in range(len(cur_lrecord)):
                            #     if cur_lrecord[i][1] in remained_fields:
                            #         new_cur_lrecord.append(cur_lrecord[i])
                            # cur_rrecord = rrecord_list[rindex]
                            # new_cur_rrecord = []
                            # for i in range(len(cur_rrecord)):
                            #     if cur_rrecord[i][1] in remained_fields:
                            #         new_cur_rrecord.append(cur_rrecord[i])
                            # jac_sim_tuple = token_based_jaccard(new_cur_lrecord, new_cur_rrecord)
                            #cur_time[0] = time.clock()
                            jac_sim_tuple = token_based_jac_cython.token_based_jaccard_cython(
                            #jac_sim_tuple = token_based_jaccard_new(
                                lrecord_list[lindex], rrecord_list[rindex],
                                lrecord_length_list[lindex], rrecord_length_list[rindex],
                                removed_field)
                            #cur_time[1] = time.clock()
                            #for i in range(3):
                            #    jac_calc_time[i] += jac_sim_tuple[2][i]

                            if len(topK_heap) == pred_list_size:
                                hq.heappushpop(topK_heap, (jac_sim_tuple[0], lindex, rindex))
                            else:
                                hq.heappush(topK_heap, (jac_sim_tuple[0], lindex, rindex))
                            #if l_pre_event[1] in compared_dict:
                            #    compared_dict[l_pre_event[1]].add(r_index)
                            #else:
                            #    compared_dict[l_pre_event[1]] = set([r_index])
                            new_compared_dict[cur_record_pair] = jac_sim_tuple[1]
                            #cur_time[2] = time.clock()
                            #for i in range(2):
                            #    step_time[i] += (cur_time[i + 1] - cur_time[i])

        while len(lprefix_events) > 0 and lprefix_events[0][0] < rprefix_events[0][0] \
                and topK_heap[0][0] < lprefix_events[0][0] * -1:
            potential_match_set = set()
            l_pre_event = hq.heappop(lprefix_events)
            new_token = lrecord_list[l_pre_event[1]][l_pre_event[2]][0]
            if new_token in lvisited_tokens_index:
                lvisited_tokens_index[new_token].add(l_pre_event[1])
            else:
                lvisited_tokens_index[new_token] = set([l_pre_event[1]])
            for i in range(l_pre_event[2] + 1):
                token_tuple = lrecord_list[l_pre_event[1]][i]
                if token_tuple[1] == removed_field:
                    continue
                token = token_tuple[0]
                if token in inc_inverted_index:
                    potential_match_set = potential_match_set.union(inc_inverted_index[token])
            if new_token in inverted_index:
                potential_match_set = potential_match_set.union(inverted_index[new_token])
            for r_index in potential_match_set:
                #if l_pre_event[1] in compared_dict and r_index in compared_dict[l_pre_event[1]]:
                #    continue
                cur_record_pair = (l_pre_event[1], r_index)

                if cur_record_pair in candidates:
                    continue

                if cur_record_pair in new_compared_dict:
                    '''If the pair has been compared in the current iteration'''
                    continue

                if cur_record_pair in reused_compared_set:
                    continue

                total_compared_pairs += 1
                if total_compared_pairs % 100000 == 0:
                    print total_compared_pairs, topK_heap[0], lprefix_events[0][0] * -1, len(lprefix_events), len(rprefix_events)
                    print get_total_size(new_compared_dict) / (1024 * 1024), get_total_size(inverted_index) / (1024 * 1024)

                if cur_record_pair in compared_set:
                    '''The pair hasn't been compared, but compared in the upper level iteration'''
                    reused_total_pairs += 1

                    llen = lrecord_length_list[l_pre_event[1]]
                    rlen = rrecord_length_list[r_index]
                    old_result = compared_set[cur_record_pair]

                    sim_upper_bound = 1.0
                    denom = llen + rlen - old_result[0]
                    if denom > 0:
                        sim_upper_bound = old_result[0] * 1.0 / denom
                    if sim_upper_bound < topK_heap[0][0]:
                        reused_ignored_pairs += 1
                        reused_compared_set.add(cur_record_pair)
                        continue

                    old_count = old_result[0]
                    old_field_map = copy.deepcopy(old_result[1])
                    for key in old_field_map.keys():
                        if key == removed_field:
                            for matched_field in old_field_map[key]:
                                old_count -= old_field_map[key][matched_field]
                            old_field_map.pop(key)
                        else:
                            if removed_field in old_field_map[key]:
                                old_count -= old_field_map[key].pop(removed_field)
                    new_compared_dict[cur_record_pair] = (old_count, old_field_map)
                    jac_sim = old_count * 1.0 / (llen + rlen - old_count)
                    if len(topK_heap) == pred_list_size:
                        hq.heappushpop(topK_heap, (jac_sim, l_pre_event[1], r_index))
                    else:
                        hq.heappush(topK_heap, (jac_sim, l_pre_event[1], r_index))
                else:
                #if True:
                    '''The pair is not compared at all. Run the complete compair procedure'''
                    new_calculated_pairs += 1
                    #cur_time = [0, 0, 0]

                    # cur_time[0] = time.clock()
                    # cur_lrecord = lrecord_list[l_pre_event[1]]
                    # new_cur_lrecord = []
                    # for i in range(len(cur_lrecord)):
                    #     if cur_lrecord[i][1] != removed_field:
                    #         new_cur_lrecord.append(cur_lrecord[i])
                    # cur_time[1] = time.clock()
                    # cur_rrecord = rrecord_list[r_index]
                    # new_cur_rrecord = []
                    # for i in range(len(cur_rrecord)):
                    #     if cur_rrecord[i][1] != removed_field:
                    #         new_cur_rrecord.append(cur_rrecord[i])
                    # cur_time[2] = time.clock()
                    # jac_sim_tuple = token_based_jaccard(new_cur_lrecord, new_cur_rrecord)
                    # cur_time[3] = time.clock()

                    #cur_time[0] = time.clock()
                    #jac_sim_tuple = token_based_jaccard_new(
                    jac_sim_tuple = token_based_jac_cython.token_based_jaccard_cython(
                        lrecord_list[l_pre_event[1]], rrecord_list[r_index],
                        lrecord_length_list[l_pre_event[1]], rrecord_length_list[r_index],
                        removed_field)
                    #cur_time[1] = time.clock()
                    #for i in range(3):
                    #    jac_calc_time[i] += jac_sim_tuple[2][i]
                    if len(topK_heap) == pred_list_size:
                        hq.heappushpop(topK_heap, (jac_sim_tuple[0], l_pre_event[1], r_index))
                    else:
                        hq.heappush(topK_heap, (jac_sim_tuple[0], l_pre_event[1], r_index))
                    #if l_pre_event[1] in compared_dict:
                    #    compared_dict[l_pre_event[1]].add(r_index)
                    #else:
                    #    compared_dict[l_pre_event[1]] = set([r_index])
                    new_compared_dict[cur_record_pair] = jac_sim_tuple[1]

                    if jac_sim_tuple[0] > l_pre_event[0] * -1.0:
                        print l_pre_event[0]* -1.0, jac_sim_tuple[0], l_pre_event, r_index, new_token
                        print lrecord_list[l_pre_event[1]]
                        print new_cur_lrecord
                        print rrecord_list[r_index]
                        print new_cur_rrecord
                        #print inc_inverted_index
                    #cur_time[2] = time.clock()
                    #for i in range(2):
                    #    step_time[i] += (cur_time[i + 1] - cur_time[i])

        #update_start_time = time.clock()
        for inc_key in inc_inverted_index:
            if inc_key in inverted_index:
                inverted_index[inc_key] = inverted_index[inc_key].union(inc_inverted_index[inc_key])
            else:
                inverted_index[inc_key] = inc_inverted_index[inc_key].copy()
        inc_inverted_index.clear()
        #update_end_time = time.clock()
        #update_inc_index_time += (update_end_time - update_start_time)

    sim_join_time = time.clock() - sim_join_start
    basic_stat = [removed_field, total_compared_pairs, reused_total_pairs, reused_ignored_pairs, new_calculated_pairs, sim_join_time]

    print 'total_compared_pairs:', total_compared_pairs, '(' + str(reused_total_pairs + new_calculated_pairs) + ')'
    print 'reused_total_pairs:', reused_total_pairs, 'reused_ignored_pairs:', reused_ignored_pairs
    print 'new_calculated_pairs:', new_calculated_pairs
    print 'sim_join_time:', sim_join_time
    #print 'join_time:', step_time
    #print 'jac_calc_time:', jac_calc_time
    #print 'update_inc_index_time:', update_inc_index_time
    #print 'gene_inc_index_time:', gene_inc_index_time
    return topK_heap, new_compared_dict, basic_stat


def token_based_jaccard(list1, list2):
    len1 = len(list1)
    len2 = len(list2)

    map1 = {}
    for i in range(len1):
        map1[list1[i][0]] = list1[i][1]

    count = 0
    result_map = {}
    for i in range(len2):
        if list2[i][0] in map1:
            count += 1
            lfield = map1[list2[i][0]]
            rfield = list2[i][1]
            if lfield in result_map:
                if rfield in result_map[lfield]:
                    result_map[lfield][rfield] += 1
                else:
                    result_map[lfield][rfield] = 1
            else:
                result_map[lfield] = {}
                result_map[lfield][rfield] = 1

    jac_sim = count * 1.0 / (len1 + len2 - count)

    return jac_sim, (count, result_map)


def token_based_jaccard_new(list1, list2, list1_length, list2_length, removed_field):
    time_stamp = [0, 0, 0, 0]
    time_stamp[0] = time.clock()
    len1 = len(list1)
    len2 = len(list2)

    time_stamp[1] = time.clock()
    map1 = {}
    for i in range(len1):
        if list1[i][1] != removed_field:
            map1[list1[i][0]] = list1[i][1]

    time_stamp[2] = time.clock()
    count = 0
    result_map = {}
    for i in range(len2):
        if list2[i][1] == removed_field:
            continue
        if list2[i][0] in map1:
            count += 1
            lfield = map1[list2[i][0]]
            rfield = list2[i][1]
            if lfield in result_map:
                if rfield in result_map[lfield]:
                    result_map[lfield][rfield] += 1
                else:
                    result_map[lfield][rfield] = 1
            else:
                result_map[lfield] = {}
                result_map[lfield][rfield] = 1

    jac_sim = count * 1.0 / (list1_length + list2_length - count)
    time_stamp[3] = time.clock()
    ret_time_stamp = [0, 0, 0]
    for i in range(3):
        ret_time_stamp[i] = time_stamp[i + 1] - time_stamp[i]

    return jac_sim, (count, result_map), ret_time_stamp


def calc_record_length(record_list, remained_fields):
    average_length = 0
    record_length_list = []
    for i in range(len(record_list)):
        full_length = len(record_list[i])
        actual_length = 0
        for j in range(full_length):
            if record_list[i][j][1] in remained_fields:
                actual_length += 1
        average_length += actual_length
        record_length_list.append(actual_length)

    return record_length_list, average_length * 1.0 / len(record_list)


def generate_prefix_events(record_list, record_length_list, prefix_events, remained_fields, lower_bound):
    for i in range(len(record_list)):
        actual_length = record_length_list[i]
        full_length = len(record_list[i])
        count = 0
        for j in range(full_length):
            if record_list[i][j][1] in remained_fields :
                threshold = calc_threshold(count, actual_length)
                if threshold >= lower_bound:
                    hq.heappush(prefix_events, (-1.0 * threshold, i, j, count, record_list[i][j][1], record_list[i][j]))
                count += 1


def calc_threshold(token_index, record_length):
    return 1 - token_index * 1.0 / record_length


def sort_record_tokens_by_global_order(record_list, order_dict):
    for i in range(len(record_list)):
        #tmp_record = [(token, order_dict[token]) for token in record_list[i]]
        #tmp_record = sorted(tmp_record, key=get_token_order)
        #record_list[i] = [tup[0] for tup in tmp_record]
        record_list[i] = sorted(record_list[i], key=lambda x: order_dict[x[0]])


def build_global_token_order(record_list, order_dict):
    for record in record_list:
        for token_tuple in record:
            token = token_tuple[0]
            if order_dict.has_key(token):
                order_dict[token] = order_dict[token] + 1
            else:
                order_dict[token] = 1


def index_candidate_set(candidate_set, lrecord_id_to_index_map, rrecord_id_to_index_map, ltable_key, rtable_key):
    new_formatted_candidate_set = set([])
    #for i in range(len(candidate_set)):
    #    pair = candidate_set.ix[i]
    #    new_formatted_candidate_set.add(
    #        (lrecord_id_to_index_map[pair[ltable_key]], rrecord_id_to_index_map[pair[rtable_key]]))
    pair_list = []
    ltable_key_data = list(candidate_set[ltable_key])
    for i in range(len(ltable_key_data)):
        pair_list.append([lrecord_id_to_index_map[str(ltable_key_data[i])]])
    rtable_key_data = list(candidate_set[rtable_key])
    for i in range(len(rtable_key_data)):
        pair_list[i].append(rrecord_id_to_index_map[str(rtable_key_data[i])])
    for i in range(len(pair_list)):
        if len(pair_list[i]) != 2:
            raise Exception('Error in indexing candidate set: pair should have two values')
        new_formatted_candidate_set.add((pair_list[i][0], pair_list[i][1]))

    return new_formatted_candidate_set


def get_tokenized_column(column):
    column_token_list = []
    for value in list(column):
        tmp_value = replace_nan_to_empty(value)
        if tmp_value != '':
            tmp_list = list(tmp_value.lower().split(' '))
            column_token_list.append(tmp_list)
        else:
            column_token_list.append([''])
    return column_token_list


def get_tokenized_table_new(table, table_key, feature_list):
    record_list = []
    record_id_to_index = {}
    id_col = list(table[table_key])
    for i in range(len(id_col)):
        id_col[i] = str(id_col[i])
        if id_col[i] in record_id_to_index:
            raise Exception('record_id is already in recrd_id_to_index')
        record_id_to_index[id_col[i]] = i

    columns = table.columns[feature_list]
    print columns
    tmp_table = []
    for col in columns:
        column_token_list = get_tokenized_column(table[col])
        tmp_table.append(column_token_list)

    num_records = len(table[table_key])
    for i in range(num_records):
        token_list = []
        index_map = {}

        for j in range(len(columns)):
            tmp_col_tokens = tmp_table[j][i]
            for token in tmp_col_tokens:
                if token != '':
                    if token in index_map:
                        token_list.append((token + '_' + str(index_map[token]), j))
                        index_map[token] += 1
                    else:
                        token_list.append((token, j))
                        index_map[token] = 1
        record_list.append(token_list)

    return record_list, record_id_to_index


def get_tokenized_record(record, feature_list):
    token_list = []
    index_map = {}
    field_count = 0

    for field in record[feature_list]:
        tmp_field = replace_nan_to_empty(field)
        '''TODO(hanli): we should remove the punctuation at the end of each token'''
        if tmp_field != '':
            tmp_list = list(tmp_field.lower().split(' '))
            for token in tmp_list:
                if token != '':
                    if token in index_map:
                        token_list.append((token + '_' + str(index_map[token]), field_count))
                        index_map[token] += 1
                    else:
                        token_list.append((token, field_count))
                        index_map[token] = 1
        field_count += 1

    return token_list


def get_tokenized_table(table, feature_list):
    record_list = []
    for i in range(len(table)):
        record = table.ix[i]
        record_list.append(get_tokenized_record(record, feature_list))
    return record_list


def get_filtered_table(ltable, rtable, corres_list):
    ltable_cols = [col_pair[0] for col_pair in corres_list]
    rtable_cols = [col_pair[1] for col_pair in corres_list]
    l_mtable = MTable(ltable[ltable_cols], key=ltable.get_key())
    r_mtable = MTable(rtable[rtable_cols], key=rtable.get_key())
    return l_mtable, r_mtable


def build_col_name_index_dict(table):
    col_dict = {}
    col_names = list(table.columns)
    for i in range(len(col_names)):
        col_dict[col_names[i]] = i
    return col_dict

def filter_corres_list(ltable, rtable, ltable_col_dict, rtable_col_dict, corres_list):
    ltable_dtypes = list(ltable.dtypes)
    ltable_key = ltable.get_key()
    rtable_dtypes = list(rtable.dtypes)
    rtable_key = rtable.get_key()
    for i in reversed(range(len(corres_list))):
        lcol_name = corres_list[i][0]
        rcol_name = corres_list[i][1]
        # Filter the pair where both fields are numeric types.
        if ltable_dtypes[ltable_col_dict[lcol_name]] != numpy.dtype('O') and rtable_dtypes[rtable_col_dict[rcol_name]] != numpy.dtype('O'):
            if lcol_name != ltable_key and rcol_name != rtable_key:
                corres_list.pop(i)

    if len(corres_list) == 0:
        raise StandardError('The field correspondence list is empty after filtering: nothing could be done!')


# If the user provides the fields correspondence list, check if each field is in the corresponding table.
def check_input_field_correspondence_list(ltable, rtable, field_corres_list):
    if field_corres_list is None:
        return
    true_ltable_fields = list(ltable.columns)
    true_rtable_fields = list(rtable.columns)
    given_ltable_fields = [field[0] for field in field_corres_list]
    given_rtable_fields = [field[1] for field in field_corres_list]
    for given_field in given_ltable_fields:
        if given_field not in true_ltable_fields:
            raise StandardError('Error in checking user input field correspondence: the field \'%s\' is not in the ltable!' %(given_field))
    for given_field in given_rtable_fields:
        if given_field not in true_rtable_fields:
            raise StandardError('Error in checking user input field correspondence: the field \'%s\' is not in the rtable!' %(given_field))
    return


def get_field_correspondence_list(ltable, rtable, field_corres_list):
    corres_list = []
    if field_corres_list is None or len(field_corres_list) == 0:
        corres_list = mg.get_attr_corres(ltable, rtable)['corres']
        if len(corres_list) == 0:
            raise StandardError('Error: the field correspondence list is empty. Nothing can be done!')
    else:
        #corres_list = field_corres_list
        for tu in field_corres_list:
            corres_list.append(tu)

    return corres_list


def replace_nan_to_empty(field):
    if pd.isnull(field):
        return ''
    elif type(field) in [float, numpy.float64, int, numpy.int64]:
        return str('{0:.0f}'.format(field))
    else:
        return str(field)


def get_feature_weight(table):
    num_records = len(table)
    if num_records == 0:
        raise StandardError('Error: empty table!')
    weight = []
    for col in table.columns:
        value_set = set()
        non_empty_count = 0;
        col_values = table[col]
        for value in col_values:
            if not pd.isnull(value):
                value_set.add(value)
                non_empty_count += 1
        selectivity = 0.0
        if non_empty_count != 0:
            selectivity = len(value_set) * 1.0 / non_empty_count
        non_empty_ratio = non_empty_count * 1.0 / num_records
        weight.append(non_empty_ratio + selectivity)
    return weight


def select_features(ltable, rtable):
    lcolumns = ltable.columns
    rcolumns = rtable.columns
    lkey = ltable.get_key()
    rkey = rtable.get_key()
    lkey_index = -1
    rkey_index = -1
    if len(lcolumns) != len(rcolumns):
        raise StandardError('Error: FILTERED ltable and FILTERED rtable have different number of fields!')
    for i in range(len(lcolumns)):
        if lkey == lcolumns[i]:
            lkey_index = i
    if lkey_index < 0:
        raise StandardError('Error: cannot find key in the FILTERED ltable schema!')
    for i in range(len(rcolumns)):
        if rkey == rcolumns[i]:
            rkey_index = i
    if rkey_index < 0:
        raise StandardError('Error: cannot find key in the FILTERED rtable schema!')

    lweight = get_feature_weight(ltable)
    #logging.info('\nFinish calculate ltable feature weights.')
    rweight = get_feature_weight(rtable)
    #logging.info('\nFinish calculate rtable feature weights.')
    if len(lweight) != len(rweight):
        raise StandardError('Error: ltable and rtable don\'t have the same schema')

    Rank = namedtuple('Rank', ['index', 'weight'])
    rank_list = []
    for i in range(len(lweight)):
        rank_list.append(Rank(i, lweight[i] * rweight[i]))
    if lkey_index == rkey_index:
        rank_list.pop(lkey_index)
    else:
        # Make sure we remove the index with larger value first!!!
        if lkey_index > rkey_index:
            rank_list.pop(lkey_index)
            rank_list.pop(rkey_index)
        else:
            rank_list.pop(rkey_index)
            rank_list.pop(lkey_index)

    rank_list = sorted(rank_list, key=attrgetter('weight'), reverse=True)
    rank_index_list = []
    num_selected_fields = 0
    if len(rank_list) <= 3:
        num_selected_fields = len(rank_list)
    elif len(rank_list) <= 5:
        num_selected_fields = 3
    else:
        num_selected_fields = len(rank_list) / 2
    for i in range(num_selected_fields):
        rank_index_list.append(rank_list[i].index)
    return sorted(rank_index_list)


def get_potential_match_set(kgram_set, inverted_index):
    sets = map(lambda x: inverted_index[x], kgram_set)
    return set.union(*sets)


def get_total_size(o, handlers={}):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def main():
    #ltable = mg.read_csv('../datasets/magellan_builtin_test/table_A.csv', key='ID')
    #rtable = mg.read_csv('../datasets/magellan_builtin_test/table_B.csv', key='ID')
    #ltable = mg.read_csv('../datasets/books_test/bowker_final_custom_id.csv', key='id')
    #rtable = mg.read_csv('../datasets/books_test/walmart_final_custom_id.csv', key='id')
    #ltable = mg.read_csv('../datasets/books_full/bowker_final_custom_id.csv', key='id')
    #rtable = mg.read_csv('../datasets/books_full/walmart_final_custom_id.csv', key='id')
    #ltable = mg.read_csv('../datasets/products_test/walmart_final_custom_id_lowercase.csv', key='id')
    #rtable = mg.read_csv('../datasets/products_test/amazon_final_custom_id_python_lowercase.csv', key='id')
    #ltable = mg.read_csv('../datasets/books/BN.csv', key='id')
    #rtable = mg.read_csv('../datasets/books/amazon_books.csv', key='id')
    #ltable = mg.read_csv('../datasets/books/BN_small.csv', key='id')
    #rtable = mg.read_csv('../datasets/books/amazon_books_small.csv', key='id')
    #blocker = mg.AttrEquivalenceBlocker()
    #candidate_set = blocker.block_tables(ltable, rtable, 'pubYear', 'pubYear')

    '''CS784 datasets'''
    #ltable = mg.read_csv('../datasets/CS784/M_ganz/tableA.csv', key='id')
    #rtable = mg.read_csv('../datasets/CS784/M_ganz/tableB.csv', key='id')
    #candidate_set = mg.read_csv('../datasets/CS784/M_ganz/tableC.csv', key='_id', ltable=ltable, rtable=rtable)
    #outdir = '../datasets/CS784/M_ganz/topK_results/'
    dataset = 'S_hanli'
    ltable = mg.read_csv('../datasets/CS784/' + dataset + '/tableA.csv', key='id')
    rtable = mg.read_csv('../datasets/CS784/' + dataset + '/tableB.csv', key='id')
    candidate_set = mg.read_csv('../datasets/CS784/' + dataset + '/tableC.csv', key='_id', ltable=ltable, rtable=rtable)
    outdir = '../datasets/CS784/' + dataset + '/topK_results/'
    pred_table = iterative_topK_debug_blocker(ltable, rtable, candidate_set, outdir)

if __name__ == "__main__":
    main()