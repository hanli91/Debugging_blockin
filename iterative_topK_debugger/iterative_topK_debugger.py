#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heapq as hq
import numpy
import pandas as pd
import functools

import time

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import magellan as mg
from magellan.core.mtable import MTable

from collections import namedtuple
from collections import defaultdict
from operator import attrgetter


def iterative_topK_debug_blocker(ltable, rtable, candidate_set, pred_list_size=100, field_corres_list=None):
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

    # Build field_index_list for removing field
    field_index_list = range(len(feature_list))

    ltable_key = candidate_set.get_property('foreign_key_ltable')
    rtable_key = candidate_set.get_property('foreign_key_rtable')

    ###indexed_candidate_set = candidate_set.set_index([rtable_key, ltable_key], drop=False)
    ###candidate_index_key_set = set(indexed_candidate_set[rtable_key])
    #print candidate_index_key_set

    logging.info('\nTokenizing records')
    # Generate record lists
    lrecord_list = get_tokenized_table(ltable_filtered, feature_list)
    rrecord_list = get_tokenized_table(rtable_filtered, feature_list)

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

    topK_heap, compared_dict = run_iteration(lrecord_list, rrecord_list, {}, 0, None, pred_list_size, -1, field_index_list)
    topK_heap = sorted(topK_heap, key=lambda tup: tup[0], reverse=True)
    #for tuple in topK_heap:
        #print tuple, list(ltable.ix[tuple[1]]), list(rtable.ix[tuple[2]])
    #    print tuple, list(lrecord_list[tuple[1]]), list(rrecord_list[tuple[2]])

    #for field_index in field_index_list:
    #    remained_fields = list(field_index_list)
    #    remained_fields.remove(field_index)
    #    run_iteration(lrecord_list, rrecord_list, compared_dict, 0, None, pred_list_size, field_index, remained_fields)

    return None


def run_iteration(lrecord_list, rrecord_list,
                  compared_set, lower_bound, candidates, pred_list_size, removed_field, remained_fields):
    logging.info(('\nRecaculating the updated length of records'))
    lrecord_length_list = calc_record_length(lrecord_list, remained_fields)
    rrecord_length_list = calc_record_length(rrecord_list, remained_fields)

    logging.info('\nGenerating prefix events')
    #Generate prefix events
    lprefix_events = []
    rprefix_events = []
    generate_prefix_events(lrecord_list, lrecord_length_list, lprefix_events, remained_fields, lower_bound)
    generate_prefix_events(rrecord_list, rrecord_length_list, rprefix_events, remained_fields, lower_bound)
    print removed_field, len(lprefix_events), len(rprefix_events)

    logging.info('\n>>>Performing sim join>>>')
    topK_heap = [(-1, -1, -1)]
    topK_heap, new_compared_dict = perform_sim_join(lrecord_list, rrecord_list, lrecord_length_list, rrecord_length_list,
                                 lprefix_events, rprefix_events, compared_set, candidates,
                                 topK_heap, pred_list_size, removed_field, remained_fields)
    logging.info('\n<<<Finishing sim join<<<')
    topK_heap = sorted(topK_heap, key=lambda tup: tup[0], reverse=True)
    #for tuple in topK_heap:
        #print tuple, list(ltable.ix[tuple[1]]), list(rtable.ix[tuple[2]])
    #    print tuple, list(lrecord_list[tuple[1]]), list(rrecord_list[tuple[2]])

    return topK_heap, new_compared_dict


def perform_sim_join(lrecord_list, rrecord_list, lrecord_length_list, rrecord_length_list, lprefix_events, rprefix_events,
                     compared_set, candidates, topK_heap, pred_list_size, removed_field, remained_fields):
    inverted_index = {}

    # The compared dict for this new iteration
    new_compared_dict = {}

    lvisited_tokens_index = {}
    compared_pairs = 0
    ignored_pairs = 0
    reused_pairs = 0
    new_calculated_pairs = 0
    step_time = [0, 0, 0, 0]
    while len(lprefix_events) > 0 and topK_heap[0][0] < lprefix_events[0][0] * -1 :
    #while len(lprefix_events) > 0 and -100 < lprefix_events[0][0] * -1 :
        #print topK_heap[0], lprefix_events[0][0] * -1
        '''TODO(hanli): should consider rprefix_events size 0'''
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
                        if cur_record_pair in new_compared_dict:
                            continue

                        if cur_record_pair in compared_set:
                            '''The pair hasn't been compared, but compared in the upper level iteration'''
                            reused_pairs += 1

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
                                ignored_pairs += 1
                                continue

                            old_count = old_result[0]
                            old_field_map = old_result[1].copy()
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
                            '''The pair is not compared at all. Run the complete compair procedure'''
                            new_calculated_pairs += 1

                            cur_lrecord = lrecord_list[lindex]
                            new_cur_lrecord = []
                            for i in range(len(cur_lrecord)):
                                if cur_lrecord[i][1] in remained_fields:
                                    new_cur_lrecord.append(cur_lrecord[i])
                            cur_rrecord = rrecord_list[rindex]
                            new_cur_rrecord = []
                            for i in range(len(cur_rrecord)):
                                if cur_rrecord[i][1] in remained_fields:
                                    new_cur_rrecord.append(cur_rrecord[i])

                            jac_sim_tuple = token_based_jaccard(new_cur_lrecord, new_cur_rrecord)
                            if len(topK_heap) == pred_list_size:
                                hq.heappushpop(topK_heap, (jac_sim_tuple[0], lindex, rindex))
                            else:
                                hq.heappush(topK_heap, (jac_sim_tuple[0], lindex, rindex))
                            #if l_pre_event[1] in compared_dict:
                            #    compared_dict[l_pre_event[1]].add(r_index)
                            #else:
                            #    compared_dict[l_pre_event[1]] = set([r_index])
                            new_compared_dict[cur_record_pair] = jac_sim_tuple[1]
                        compared_pairs += 1

        while len(lprefix_events) > 0 and lprefix_events[0][0] < rprefix_events[0][0]:
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
                if cur_record_pair in new_compared_dict:
                    '''If the pair has been compared in the current iteration'''
                    continue

                if cur_record_pair in compared_set:
                    '''The pair hasn't been compared, but compared in the upper level iteration'''
                    reused_pairs += 1

                    llen = lrecord_length_list[l_pre_event[1]]
                    rlen = rrecord_length_list[r_index]
                    old_result = compared_set[cur_record_pair]

                    sim_upper_bound = 1.0
                    denom = llen + rlen - old_result[0]
                    if denom > 0:
                        sim_upper_bound = old_result[0] * 1.0 / denom
                    if sim_upper_bound < topK_heap[0][0]:
                        ignored_pairs += 1
                        continue

                    old_count = old_result[0]
                    old_field_map = old_result[1].copy()
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
                    cur_time = [0, 0, 0, 0, 0]

                    cur_time[0] = time.clock()
                    cur_lrecord = lrecord_list[l_pre_event[1]]
                    new_cur_lrecord = []
                    for i in range(len(cur_lrecord)):
                        if cur_lrecord[i][1] != removed_field:
                            new_cur_lrecord.append(cur_lrecord[i])
                    cur_time[1] = time.clock()
                    cur_rrecord = rrecord_list[r_index]
                    new_cur_rrecord = []
                    for i in range(len(cur_rrecord)):
                        if cur_rrecord[i][1] != removed_field:
                            new_cur_rrecord.append(cur_rrecord[i])
                    cur_time[2] = time.clock()
                    jac_sim_tuple = token_based_jaccard(new_cur_lrecord, new_cur_rrecord)
                    cur_time[3] = time.clock()
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
                    cur_time[4] = time.clock()
                    for i in range(4):
                        step_time[i] += (cur_time[i + 1] - cur_time[i])

                compared_pairs += 1


        for inc_key in inc_inverted_index:
            if inc_key in inverted_index:
                inverted_index[inc_key] = inverted_index[inc_key].union(inc_inverted_index[inc_key])
            else:
                inverted_index[inc_key] = inc_inverted_index[inc_key].copy()
        inc_inverted_index.clear()


    print 'compared:', compared_pairs, 'ignored:', ignored_pairs, compared_pairs + ignored_pairs
    print 'reused:', reused_pairs, 'new_calculated:', new_calculated_pairs, reused_pairs + new_calculated_pairs
    print step_time
    return topK_heap, new_compared_dict


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


def calc_record_length(record_list, remained_fields):
    record_length_list = []
    for i in range(len(record_list)):
        full_length = len(record_list[i])
        actual_length = 0
        for j in range(full_length):
            if record_list[i][j][1] in remained_fields:
                actual_length += 1
        record_length_list.append(actual_length)

    return record_length_list


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


def main():
    #ltable = mg.read_csv('../datasets/magellan_builtin_test/table_A.csv', key='ID')
    #rtable = mg.read_csv('../datasets/magellan_builtin_test/table_B.csv', key='ID')
    ltable = mg.read_csv('../datasets/books_test/bowker_final_custom_id.csv', key='id')
    rtable = mg.read_csv('../datasets/books_test/walmart_final_custom_id.csv', key='id')
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
    candidate_set = MTable()
    pred_table = iterative_topK_debug_blocker(ltable, rtable, candidate_set)

if __name__ == "__main__":
    main()