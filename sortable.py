"""Shared client-side snippet that makes every <table> in a report sortable.

Inject the output of `sortable_html()` immediately before `</body></html>` in
any generated report.  The script:

  * Auto-discovers every <table> with a <thead>.
  * Adds click handlers to the *last* row of <thead> (so super-column headers
    spanning multiple columns don't get treated as data columns).
  * Sorts the first <tbody> in ascending / descending order, toggling on each
    click.  Numeric cells (.327, -186, +24, "30.5%", "0.408 → 0.471",
    "3: .165 / .302 / .540") sort by their first numeric value; everything
    else falls back to case-insensitive string sort.
  * Skips any <table> or <th> tagged with the `no-sort` CSS class.
  * Renders a faint up/down arrow on each sortable header, brightened when
    that column is the active sort.

Cells with a `data-sort-value` attribute override the parsed value, which is
handy for columns where the displayed text is a label but the underlying
quantity is numeric (e.g. "DNP" sorting last).
"""

from __future__ import annotations

_SORTABLE_CSS = """
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { background: rgba(0, 0, 0, 0.05); }
th.sortable::after { content: ' \\2195'; opacity: 0.25; font-size: 0.85em;
                     font-weight: 400; margin-left: 2px; }
th.sortable[data-sort="asc"]::after  { content: ' \\25B2'; opacity: 0.9; }
th.sortable[data-sort="desc"]::after { content: ' \\25BC'; opacity: 0.9; }
"""

_SORTABLE_JS = r"""
(function () {
  function parseCell(cell) {
    var override = cell.getAttribute('data-sort-value');
    var raw = override != null ? override : (cell.innerText || cell.textContent || '');
    var s = raw.trim();
    if (s === '' || s === '-' || s === '\u2014') {
      return { n: NaN, s: '', empty: true };
    }
    var m = s.match(/-?\+?\d+(?:\.\d+)?/);
    var n = m ? parseFloat(m[0]) : NaN;
    return { n: n, s: s.toLowerCase(), empty: false };
  }
  function compare(a, b, dir) {
    var pa = parseCell(a), pb = parseCell(b);
    if (pa.empty && pb.empty) return 0;
    if (pa.empty) return 1;   // empties always sink
    if (pb.empty) return -1;
    var aNum = !isNaN(pa.n), bNum = !isNaN(pb.n);
    if (aNum && bNum) return dir * (pa.n - pb.n);
    if (aNum) return -1;      // numbers before text
    if (bNum) return 1;
    return dir * pa.s.localeCompare(pb.s);
  }
  function sortTable(table, col, dir) {
    var tbody = table.tBodies[0];
    if (!tbody) return;
    var withCell = [], without = [];
    Array.prototype.forEach.call(tbody.rows, function (r) {
      if (r.cells[col]) withCell.push(r); else without.push(r);
    });
    withCell.sort(function (a, b) { return compare(a.cells[col], b.cells[col], dir); });
    withCell.forEach(function (r) { tbody.appendChild(r); });
    without.forEach(function (r) { tbody.appendChild(r); });
  }
  function attach(table) {
    var thead = table.tHead;
    if (!thead || !thead.rows.length) return;
    var hrow = thead.rows[thead.rows.length - 1];
    Array.prototype.forEach.call(hrow.cells, function (th, idx) {
      if (th.classList.contains('no-sort')) return;
      th.classList.add('sortable');
      th.addEventListener('click', function () {
        var current = th.getAttribute('data-sort') || 'none';
        Array.prototype.forEach.call(hrow.cells, function (other) {
          if (other !== th) other.removeAttribute('data-sort');
        });
        var next = current === 'asc' ? 'desc' : 'asc';
        th.setAttribute('data-sort', next);
        sortTable(table, idx, next === 'asc' ? 1 : -1);
      });
    });
  }
  function init() {
    Array.prototype.forEach.call(document.querySelectorAll('table'), function (t) {
      if (t.classList.contains('no-sort')) return;
      attach(t);
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
"""


def sortable_html() -> str:
    """Return a `<style>` + `<script>` block to inject before `</body>`."""
    return f"<style>{_SORTABLE_CSS}</style>\n<script>{_SORTABLE_JS}</script>"


def sortable_style() -> str:
    """Just the CSS (use when you want to merge into an existing <style>)."""
    return _SORTABLE_CSS


def sortable_script() -> str:
    """Just the `<script>...</script>` tag."""
    return f"<script>{_SORTABLE_JS}</script>"
