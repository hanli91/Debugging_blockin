import time
from array import array

def token_based_jaccard_cython(list1, list2, int list1_length, int list2_length, int removed_field, int offset_of_field_num):
    cdef int len1 = len(list1)
    cdef int len2 = len(list2)
    cdef int i, j

    map1 = {}
    for i in range(len1):
        if list1[i][1] != removed_field:
            map1[list1[i][0]] = list1[i][1]

    #cdef int count = 0
    #result_map = {}
    #for i in range(len2):
    #    if list2[i][1] == removed_field:
    #        continue
    #    if list2[i][0] in map1:
    #        count += 1
    #        lfield = map1[list2[i][0]]
    #        rfield = list2[i][1]
    #        if lfield in result_map:
    #            if rfield in result_map[lfield]:
    #                result_map[lfield][rfield] += 1
    #            else:
    #                result_map[lfield][rfield] = 1
    #        else:
    #            result_map[lfield] = {}
    #            result_map[lfield][rfield] = 1

    cdef int count = 0
    result_map = {}
    for i in range(len2):
        if list2[i][1] == removed_field:
            continue
        if list2[i][0] in map1:
            count += 1
            lfield = map1[list2[i][0]]
            rfield = list2[i][1]
            tmp = lfield * offset_of_field_num + rfield
            if tmp in result_map:
                result_map[tmp] += 1
            else:
                result_map[tmp] = 1

    ret_list = array('I', [])
    for key in result_map:
        ret_list.append(result_map[key] * offset_of_field_num * offset_of_field_num + key)
    del result_map

    cdef double jac_sim = count * 1.0 / (list1_length + list2_length - count)

    return jac_sim, (count, ret_list)


def update_old_field_matching_list(int old_count, old_list, remained_fields, int offset_of_field_num):
    cdef int i, tmp
    cdef int length = len(old_list)
    cdef int count, lfield, rfield
    updated_list = array('I', [])
    for i in range(length):
        tmp = old_list[i] / offset_of_field_num
        rfield = old_list[i] % offset_of_field_num
        lfield = tmp % offset_of_field_num
        if lfield in remained_fields and rfield in remained_fields:
            updated_list.append(old_list[i])
        else:
            old_count -= tmp / offset_of_field_num

    return old_count, updated_list