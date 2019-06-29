# *-* coding: utf-8 *-*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def lazy_map(tokens):
    ids = []
    mark_dict = dict()
    for token in tokens:
        if token not in mark_dict:
            mark_dict[token] = len(mark_dict)
        ids.append[mark_dict]
    return ids


class BasicIdMap(object):
    """可增token到id的双射
    """
    def __init__(self, tokens=None, is_unique=False, default='<unk>'):
        self._token_to_id = dict()
        self._id_to_token = list()

        if tokens:
            if is_unique:
                self._token_to_id = dict(zip(tokens, range(len(tokens))))
                self._id_to_token = tokens
            else:
                self.extend_tokens(tokens)

        self.add_token(default)
        self._default_token = default
        self._default_id = self.get_id(default)

    def __len__(self):
        return len(self._id_to_token)

    @property
    def vocab(self):
        return self._id_to_token

    @property
    def blind_token(self):
        return self._default_token

    @property
    def blind_id(self):
        return self._default_id

    def has_token(self, token):
        return (token in self._token_to_id)

    def extend_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self._id_to_token)
            self._id_to_token.append(token)

    def get_id(self, token):
        return self._token_to_id[token] if token in self._token_to_id\
            else self._default_id

    def get_token(self, i):
        return self._id_to_token[i] if i < len(self._id_to_token)\
            else self._default_token
