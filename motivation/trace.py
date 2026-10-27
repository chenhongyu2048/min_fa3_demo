"""Decode CTA records without treating different GPUs as a shared clock."""

from .config import PHASES


def decode_trace(trace, sm_count):
    if len(trace) != len(PHASES):
        raise ValueError("expected six phases")
    phases = []
    previous_exit = {}
    previous_sms = None
    for phase, rows in enumerate(trace):
        if len(rows) != sm_count or any(len(row) != 7 for row in rows):
            raise ValueError("expected [6, num_sms, 7] trace")
        sms = [row[0] for row in rows]
        # Physical SM IDs need not be contiguous on a harvested GPU.
        if len(set(sms)) != sm_count or any(sm < 0 for sm in sms):
            raise ValueError("trace does not cover one distinct SM per CTA")
        if previous_sms is not None and sms != previous_sms:
            raise ValueError("CTA moved to a different SM within the persistent kernel")
        previous_sms = sms
        last_work_end = max(row[3] for row in rows)
        first_start = min(row[2] for row in rows)
        decoded = []
        for cta, (sm, entry, start, end, exit_time, rank_in, rank_out) in enumerate(rows):
            if not (0 < entry <= start <= end <= exit_time):
                raise ValueError("nonmonotonic phase timestamps")
            if entry < previous_exit.get(cta, 0):
                raise ValueError("next phase precedes prior phase exit")
            if exit_time < last_work_end:
                raise ValueError("grid barrier released before all CTA work ended")
            if cta == 0 and phase in (0, 4):
                if not entry <= rank_in <= rank_out <= start:
                    raise ValueError("invalid communication rank barrier interval")
            elif rank_in != 0 or rank_out != 0:
                raise ValueError("unexpected rank barrier record")
            previous_exit[cta] = exit_time
            decoded.append({"cta": cta, "sm": sm, "entry_ns": entry, "start_ns": start,
                            "end_ns": end, "exit_ns": exit_time,
                            "entry_wait_ns": start - entry, "work_ns": end - start,
                            "tail_idle_ns": last_work_end - end,
                            "exit_wait_ns": exit_time - end})
        phases.append({"phase": PHASES[phase], "ctas": decoded,
                       "work_span_ns": last_work_end - first_start,
                       "tail_idle_sm_ns": sum(row["tail_idle_ns"] for row in decoded),
                       "rank_wait_ns": rows[0][6] - rows[0][5]})
    return phases
