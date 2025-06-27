import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
from ttkthemes import ThemedTk
import hashlib, time, json, sqlite3
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =============================================================================
# Supply Chain Blockchain Manager v2.0
# Features: SQLite persistence, advanced GUI, embedded visualizations
# =============================================================================

# === Supply Chain Stages ===
SUPPLY_CHAIN_STAGES = [
    "Farm", "Storage", "Processing", "Manufacturing",
    "Distribution", "Retailer", "Consumer"
]

# === Data Models ===
class SupplyTransaction:
    def __init__(self, product_id, stage, timestamp=None):
        self.product_id = product_id
        self.stage = stage
        self.timestamp = timestamp or time.time()

    def to_dict(self):
        return {
            "product_id": self.product_id,
            "stage": self.stage,
            "timestamp": self.timestamp
        }

    @staticmethod
    def from_dict(data):
        return SupplyTransaction(
            data["product_id"],
            data["stage"],
            data["timestamp"]
        )

class Block:
    def __init__(self, index, prev_hash, transactions, nonce=0, timestamp=None):
        self.index = index
        self.prev_hash = prev_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.nonce = nonce
        self.hash = self.calc_hash()

    def calc_hash(self):
        data = {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "nonce": self.nonce,
            "timestamp": self.timestamp
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "nonce": self.nonce,
            "timestamp": self.timestamp
        }

    @staticmethod
    def from_dict(data):
        txs = [SupplyTransaction.from_dict(t) for t in data.get("transactions", [])]
        return Block(
            data.get("index"),
            data.get("prev_hash"),
            txs,
            data.get("nonce", 0),
            data.get("timestamp")
        )

class SupplyBlockchain:
    def __init__(self, db_path="blockchain.db", difficulty=3):
        self.difficulty = difficulty
        self.chain = []
        self.products = {}
        self.db = sqlite3.connect(db_path)
        self._init_db()
        self._load_chain()

    def _init_db(self):
        c = self.db.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                idx INTEGER PRIMARY KEY,
                prev_hash TEXT,
                hash TEXT,
                nonce INTEGER,
                timestamp REAL
            )
        """ )
        c.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                pid TEXT,
                stage TEXT,
                timestamp REAL,
                block_idx INTEGER,
                FOREIGN KEY(block_idx) REFERENCES blocks(idx)
            )
        """ )
        self.db.commit()

    def _load_chain(self):
        c = self.db.cursor()
        c.execute(
            "SELECT idx, prev_hash, hash, nonce, timestamp FROM blocks ORDER BY idx"
        )
        rows = c.fetchall()
        for idx, prev_hash, h, nonce, ts in rows:
            c.execute(
                "SELECT pid, stage, timestamp FROM transactions WHERE block_idx=?", (idx,)
            )
            txs = [SupplyTransaction(pid, stage, timestamp)
                   for pid, stage, timestamp in c.fetchall()]
            blk = Block(idx, prev_hash, txs, nonce, ts)
            blk.hash = h
            self.chain.append(blk)
            for tx in txs:
                self.products.setdefault(tx.product_id, []).append(tx)
        # If no blocks, create genesis
        if not self.chain:
            genesis = Block(0, "0", [])
            self.chain.append(genesis)
            self._save_block(genesis)

    def _save_block(self, blk):
        c = self.db.cursor()
        c.execute(
            "INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?)",
            (blk.index, blk.prev_hash, blk.hash, blk.nonce, blk.timestamp)
        )
        for tx in blk.transactions:
            c.execute(
                "INSERT INTO transactions VALUES (?,?,?,?)",
                (tx.product_id, tx.stage, tx.timestamp, blk.index)
            )
        self.db.commit()

    def add_product(self, pid):
        if not pid or pid in self.products:
            return False
        self.products[pid] = []
        return True

    def update_stage(self, pid, stage):
        tx = SupplyTransaction(pid, stage)
        self.products[pid].append(tx)
        return tx

    def mine_block(self, blk, progress=None):
        target = "0" * self.difficulty
        blk.hash = blk.calc_hash()
        while not blk.hash.startswith(target):
            blk.nonce += 1
            blk.hash = blk.calc_hash()
            if progress:
                progress(blk.nonce, blk.hash)
        self.chain.append(blk)
        self._save_block(blk)

    def to_dict(self):
        return {
            "difficulty": self.difficulty,
            "chain": [b.to_dict() for b in self.chain],
            "products": {
                pid: [tx.to_dict() for tx in txs]
                for pid, txs in self.products.items()
            }
        }

    @staticmethod
    def from_dict(data, db_path="blockchain.db"):
        bc = SupplyBlockchain(db_path, data.get("difficulty", 3))
        bc.chain = [Block.from_dict(b) for b in data.get("chain", [])]
        bc.products = {
            pid: [SupplyTransaction.from_dict(t) for t in txs]
            for pid, txs in data.get("products", {}).items()
        }
        return bc

# =============================================================================
# GUI Application
# =============================================================================
class AdvancedBlockchainApp:
    def __init__(self, master):
        self.master = master
        master.title("🚀 Supply Chain Blockchain Manager v2.0")
        master.geometry("920x780")

        self.bc = SupplyBlockchain()
        self._build_menu()

        ttk.Label(
            master,
            text="Supply Chain Blockchain Manager v2.0",
            font=("Arial", 22, "bold")
        ).pack(pady=8)

        self._build_product_section()
        self._build_action_section()
        self._build_status_section()
        self._build_analytics_section()
        self._build_log_section()

    def _build_menu(self):
        menubar = tk.Menu(self.master)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save...", command=self._save)
        file_menu.add_command(label="Load...", command=self._load)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Set Difficulty", command=self._set_difficulty)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.master.config(menu=menubar)

    def _build_product_section(self):
        frm = ttk.LabelFrame(self.master, text="Products")
        frm.pack(fill='x', padx=10, pady=5)

        ttk.Label(frm, text="Product ID:").grid(row=0, column=0, padx=5, pady=5)
        self.pid_entry = ttk.Entry(frm)
        self.pid_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frm, text="Add", command=self._add_product).grid(
            row=0, column=2, padx=5
        )

        cols = ("ProductID", "Stages")
        self.tree = ttk.Treeview(
            frm, columns=cols, show='headings', height=6
        )
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=200)
        self.tree.grid(row=1, column=0, columnspan=3, pady=5, sticky='nsew')
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(1, weight=1)

    def _build_action_section(self):
        frm = ttk.LabelFrame(self.master, text="Actions")
        frm.pack(fill='x', padx=10, pady=5)

        ttk.Button(frm, text="Mine Block", command=self._mine).pack(
            side='left', padx=5, pady=5
        )
        ttk.Button(
            frm, text="Visualize Chain", command=self._embed_chain
        ).pack(side='left', padx=5)
        ttk.Button(
            frm, text="Visualize Flow", command=self._embed_flow
        ).pack(side='left', padx=5)

    def _build_status_section(self):
        frm = ttk.LabelFrame(self.master, text="Status")
        frm.pack(fill='x', padx=10, pady=5)

        self.pb = ttk.Progressbar(frm, mode='indeterminate')
        self.pb.pack(fill='x', padx=5, pady=2)
        self.nl = ttk.Label(frm, text="Nonce: N/A")
        self.nl.pack(side='left', padx=10)
        self.hl = ttk.Label(frm, text="Hash: N/A")
        self.hl.pack(side='left', padx=10)

    def _build_analytics_section(self):
        frm = ttk.LabelFrame(self.master, text="Analytics")
        frm.pack(fill='both', padx=10, pady=5, expand=True)
        ttk.Label(
            frm,
            text="Future analytics dashboards will appear here."
        ).pack(pady=20)

    def _build_log_section(self):
        frm = ttk.LabelFrame(self.master, text="Log")
        frm.pack(fill='both', expand=True, padx=10, pady=5)
        self.log = scrolledtext.ScrolledText(frm, state='disabled')
        self.log.pack(fill='both', expand=True)

    def _log(self, msg):
        self.log.config(state='normal')
        self.log.insert(tk.END, f"{time.ctime()} - {msg}\n")
        self.log.config(state='disabled')
        self.log.yview(tk.END)

    def _refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for pid, txs in self.bc.products.items():
            self.tree.insert('', tk.END, values=(pid, len(txs)))

    def _add_product(self):
        pid = self.pid_entry.get().strip()
        if self.bc.add_product(pid):
            self._log(f"Product '{pid}' added.")
            self._refresh()
        else:
            messagebox.showerror(
                "Error", "Invalid or duplicate Product ID."
            )

    def _mine(self):
        pid = self.pid_entry.get().strip()
        if pid not in self.bc.products:
            messagebox.showerror(
                "Error", "Please add a valid Product first."
            )
            return
        txs = [
            self.bc.update_stage(pid, stage)
            for stage in SUPPLY_CHAIN_STAGES
        ]
        block = Block(
            len(self.bc.chain), self.bc.chain[-1].hash, txs
        )
        self.pb.start()
        self.bc.mine_block(block, self._update_status)
        self.pb.stop()
        self._log(
            f"Mined block {block.index} with nonce={block.nonce}."
        )
        self._refresh()

    def _update_status(self, nonce, hashval):
        self.nl.config(text=f"Nonce: {nonce}")
        self.hl.config(text=f"Hash: {hashval[:16]}...")
        self.master.update()

    def _embed_chain(self):
        win = tk.Toplevel(self.master)
        win.title("Blockchain Visualization")
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111, facecolor='#eef')

        G = nx.DiGraph()
        labels = {}
        for blk in self.bc.chain:
            G.add_node(blk.index)
            labels[blk.index] = f"{blk.index}\n{blk.hash[:4]}"
            if blk.index > 0:
                G.add_edge(blk.index-1, blk.index)

        pos = nx.spring_layout(G, seed=1)
        colors = plt.cm.viridis([i/len(G.nodes) for i in G.nodes])
        nx.draw(
            G, pos, ax=ax,
            node_color=colors,
            edge_color='gray',
            with_labels=False,
            node_size=800
        )
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        ax.set_title('Blockchain Structure')

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _embed_flow(self):
        pid = self.pid_entry.get().strip()
        txs = self.bc.products.get(pid, [])
        if not txs:
            messagebox.showerror("Error", "No transactions to display.")
            return

        win = tk.Toplevel(self.master)
        win.title(f"Product Flow: {pid}")
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111, facecolor='#fee')

        times = [tx.timestamp for tx in txs]
        stages = [SUPPLY_CHAIN_STAGES.index(tx.stage) for tx in txs]
        cmap = plt.cm.plasma
        ax.scatter(
            times, stages,
            c=range(len(stages)),
            cmap=cmap,
            s=80
        )
        ax.plot(times, stages, linewidth=2)
        ax.set_yticks(range(len(SUPPLY_CHAIN_STAGES)))
        ax.set_yticklabels(SUPPLY_CHAIN_STAGES)
        ax.set_title(f'Product Flow for {pid}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stage')
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON', '*.json')]
        )
        if path:
            with open(path, 'w') as f:
                json.dump(self.bc.to_dict(), f, indent=2)
            self._log(f"Saved blockchain to {path}")

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[('JSON', '*.json')]
        )
        if path:
            with open(path) as f:
                data = json.load(f)
            self.bc = SupplyBlockchain.from_dict(data)
            self._refresh()
            self._log(f"Loaded blockchain from {path}")

    def _set_difficulty(self):
        d = simpledialog.askinteger(
            'Difficulty', 'Enter difficulty (1-6):',
            minvalue=1, maxvalue=6
        )
        if d:
            self.bc.difficulty = d
            self._log(f"Difficulty set to {d}")

    def _about(self):
        messagebox.showinfo(
            'About', 'Supply Chain Blockchain Manager v2.0'
        )

if __name__ == '__main__':
    root = ThemedTk(theme='arc')
    app = AdvancedBlockchainApp(root)
    root.mainloop()
