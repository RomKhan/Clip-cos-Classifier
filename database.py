import sqlite3
import numpy as np
import io
import os
from scipy.sparse import csr_array

class Database():
    def __init__(self, destination_path=None, mode=None):
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)
        self.destination_path = destination_path
        self.mode = mode
        self.db_conn = self.get_or_create_db()
        self.cursor = None
        self.cursor_counter = 0

    def adapt_array(self, arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def get_or_create_db(self):
        if self.mode != 'filter db':
            conn = sqlite3.connect(os.path.join(self.destination_path, 'image_embeddings.db'), detect_types=sqlite3.PARSE_DECLTYPES)
        else:
            conn = sqlite3.connect(os.path.join(self.destination_path, 'filter_images.db'), detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS offer (offer_id TEXT PRIMARY KEY, images_id array)''')
        cursor.close()
        conn.commit()
        return conn

    def get_offer_idx(self):
        cursor = self.db_conn.cursor()
        cursor.execute(f"SELECT * FROM offer")
        idx = cursor.fetchall()
        cursor.close()
        idx_dict = {}
        for pair in idx:
            idx_dict[pair[0]] = pair[1]
        return idx_dict

    def select(self, key, table, cursor=None):
        is_none = True if cursor is None else False
        if is_none:
            cursor = self.db_conn.cursor()
        cursor.execute(f"SELECT * FROM {table} WHERE {table}_id = ?", (key,))
        row = cursor.fetchone()
        if is_none:
            cursor.close()
        return row

    def select_multiple(self, key, table):
        if self.cursor is not None and self.cursor_counter > 10000:
            self.db_conn.commit()
            self.cursor.close()
            self.cursor = None
            self.cursor_counter = 0
        elif self.cursor is None:
            self.cursor = self.db_conn.cursor()
            self.cursor_counter = 0
        return self.select(key, table, self.cursor)

    # def insert_offer_images(self, images):
    #     cursor = self.db_conn.cursor()
    #     idx = []
    #     for image in images:
    #         if self.mode != 'filter db':
    #             clip_embedding, resnext_embedding, logits, path = image
    #             # clip_embedding = clip_embedding.toarray()
    #             resnext_embedding = resnext_embedding.toarray()
    #             cursor.execute("INSERT INTO image (clip_embedding, resnext_embedding, logits, path) VALUES (?, ?, ?, ?)", (clip_embedding, resnext_embedding, logits, path))
    #         else:
    #             clip_embedding, path = image
    #             cursor.execute("INSERT INTO image (clip_embedding, path) VALUES (?, ?)",(clip_embedding, path))
    #         idx.append(cursor.lastrowid)
    #     self.db_conn.commit()
    #     cursor.close()
    #     return np.array(idx, dtype='uint32')

    def insert_offers(self, offers):
        cursor = self.db_conn.cursor()
        for offer in offers:
            offer_id, images_id = offer
            cursor.execute("INSERT INTO offer (offer_id, images_id) VALUES (?, ?)",(offer_id, images_id))
        self.db_conn.commit()
        cursor.close()

    @staticmethod
    def save_relevants(path_to_relevants, clip_relevants, resnext_relevants):
        conn = sqlite3.connect(os.path.join(path_to_relevants, 'relevants.db'), detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS relevants (image_id INTEGER PRIMARY KEY, clip_relevants array, resnext_relevants array)''')
        conn.commit()
        cursor.execute("SELECT * FROM relevants")
        rows = cursor.fetchall()
        keys = list(clip_relevants.keys())
        for key in keys:
            cursor.execute("INSERT OR REPLACE INTO relevants (image_id, clip_relevants, resnext_relevants) VALUES (?, ?, ?)",(key+1, clip_relevants[key], resnext_relevants[key]))

        conn.commit()
        cursor.close()
        conn.close()











