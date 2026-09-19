"""Unit tests for src/data/repair.py on rows whose ground truth is known.

The DDoS_UDP fixtures are real CSV rows from the dataset's own
``Attack traffic/DDoS_UDP_Flood_attack.csv``; the expected values come from
tshark 4.2.2 on the shipped ``DDoS UDP Flood Attacks.pcap``:

  packet 1          "Nov 24, 2021 22:27:10.139612000 UTC|255.16.17.190|192.168.0.128|...|12912,0|0|..."
  stream 98549 ICMP "...|192.168.0.128,6.105.66.145|6.105.66.145,192.168.0.128|||||0x0739|||00000000|...|24624,0|98549|..."

The Normal fixtures are rows 1270940 (DNS) and 1248878 (NTP) of
DNN-EdgeIIoT-dataset.csv; their expected values follow from protocol semantics
(qtype 1 = A, query-name length 21 = len("0.debian.pool.ntp.org"), NTP port 123).

    python -m unittest tests.test_repair -v
"""
import math
import unittest

import pandas as pd

from src.data import repair as R

F = R.EDGE_FEATURES


def row(values, label, attack_label):
    assert len(values) == len(F), len(values)
    return dict(zip(F, values), Attack_label=attack_label, Attack_type=label)


def frame(*rows):
    return pd.DataFrame(list(rows), dtype=object)


UDP_FLOOD = row(("255.16.17.190,192.168.0.128,0,0,0.0,0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,"
                 "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"
                 "12912.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"
                 "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0").split(","), "DDoS_UDP", "1")

ICMP_REPLY_VALUES = ["0.0"] * len(F)
ICMP_REPLY_VALUES[:9] = ["192.168.0.128", "6.105.66.145", "6.105.66.145", "192.168.0.128",
                         "0.0", "0", "0.0", "0.0", "0x00000739"]
ICMP_REPLY_VALUES[36] = "24624.0"
ICMP_REPLY_VALUES[38] = "98549.0"
ICMP_REPLY = row(ICMP_REPLY_VALUES, "DDoS_UDP", "1")

NORMAL_DNS = row([" 2021 16:01:19.409015000 ", "192.168.0.128", "192.168.0.1", "0", "0.0", "0.0",
                  "0"] + ["0.0"] * 24 + ["0", "0", "0.0", "0.0", "39080.0", "53.0", "103.0",
                  "0.0", "0.debian.pool.ntp.org", "21.0", "0.0", "1.0", "0.0", "0.0", "0", "0.0",
                  "0.0", "0.0", "0.0", "0.0", "0", "0.0", "0.0", "0", "0", "0.0", "0.0", "0.0",
                  "0.0", "0.0"], "Normal", "0")

NORMAL_NTP_VALUES = list(NORMAL_DNS.values())[:len(F)]
NORMAL_NTP_VALUES[2] = "168.167.71.131"
NORMAL_NTP_VALUES[35:40] = ["60542.0", "123.0", "97.0", "0.0", "0"]
NORMAL_NTP_VALUES[40:43] = ["0.0", "0.0", "0.0"]
NORMAL_NTP = row(NORMAL_NTP_VALUES, "Normal", "0")

TCP_ROW = row([" 2021 23:17:49.1 ", "10.0.0.1", "192.168.0.128"] + ["0.0"] * 58, "DDoS_TCP", "1")
TCP_ROW["mqtt.topic"] = "0.0"
TCP_ROW["mqtt.conack.flags"] = "0x00000000"
TCP_ROW["http.referer"] = "() { _; } >_[$($())] { echo 93e4r0-CVE-2014-6278: true; echo;echo; }"
MITM = row(["6.0"] + ["0.0"] * 60, "MITM", "1")


def num(x):
    return float(x)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.raw = frame(UDP_FLOOD, ICMP_REPLY, NORMAL_DNS, NORMAL_NTP, TCP_ROW, MITM)
        self.prov = R.classify_layout(self.raw)
        self.fixed = R.realign_rows(self.raw, self.prov)

    def test_classification(self):
        self.assertEqual(list(self.prov), ["decoded_udp_flood", "decoded_icmp_reply",
                                           "decoded_normal_udp", "decoded_normal_udp",
                                           "original", "dropped_mitm"])

    def test_udp_flood_matches_pcap_packet_1(self):
        r = self.fixed.iloc[0]
        self.assertTrue(isinstance(r["frame.time"], float) and math.isnan(r["frame.time"]))
        self.assertEqual(r["ip.src_host"], "255.16.17.190")
        self.assertEqual(r["ip.dst_host"], "192.168.0.128")
        self.assertEqual(num(r["udp.port"]), 12912)       # first of "12912,0"
        self.assertEqual(num(r["udp.stream"]), 0)
        self.assertEqual(num(r["tcp.srcport"]), 0)        # no TCP on this packet
        self.assertEqual(num(r["arp.dst.proto_ipv4"]), 0)

    def test_icmp_reply_matches_pcap_stream_98549(self):
        r = self.fixed.iloc[1]
        self.assertEqual(r["ip.src_host"], "192.168.0.128")   # outer header
        self.assertEqual(r["ip.dst_host"], "6.105.66.145")
        self.assertEqual(num(r["arp.dst.proto_ipv4"]), 0)
        self.assertEqual(int(r["icmp.checksum"], 16), 0x0739)
        self.assertEqual(num(r["icmp.seq_le"]), 0)
        self.assertEqual(num(r["udp.port"]), 24624)           # inner source port
        self.assertEqual(num(r["udp.stream"]), 98549)
        self.assertEqual(num(r["dns.qry.name"]), 0)
        self.assertEqual(r["mbtcp.trans_id"], "0")            # truncated upstream
        self.assertEqual(r["mbtcp.unit_id"], "0")

    def test_normal_dns_semantics(self):
        r = self.fixed.iloc[2]
        self.assertEqual(r["ip.dst_host"], "192.168.0.1")
        self.assertEqual(num(r["udp.port"]), 39080)
        self.assertEqual(num(r["udp.stream"]), 103)
        self.assertEqual(r["dns.qry.name"], "0.debian.pool.ntp.org")
        self.assertEqual(num(r["dns.qry.name.len"]), len("0.debian.pool.ntp.org"))
        self.assertIn(num(r["dns.qry.qu"]), (0, 1))
        self.assertEqual(num(r["dns.qry.type"]), 1)           # A record
        self.assertEqual(r["mbtcp.unit_id"], "0")

    def test_normal_ntp_semantics(self):
        r = self.fixed.iloc[3]
        self.assertEqual(num(r["udp.port"]), 60542)
        self.assertEqual(num(r["udp.stream"]), 97)
        self.assertEqual(num(r["udp.time_delta"]), 0)

    def test_original_rows_untouched(self):
        pd.testing.assert_series_equal(self.fixed.iloc[4], self.raw.iloc[4])

    def test_decoded_rows_pass_validation(self):
        kept = self.fixed[(self.prov != "dropped_mitm").to_numpy()]
        self.assertFalse(R.row_violations(kept).any())

    def test_raw_rows_fail_validation(self):
        self.assertTrue(R.row_violations(self.raw.iloc[[2]]).all())   # hostname in name.len

    def test_unknown_udp_layout_raises(self):
        bad = dict(TCP_ROW, **{"udp.port": "5000.0"})
        with self.assertRaises(ValueError):
            R.classify_layout(frame(bad))

    def test_udp_row_without_address_raises(self):
        bad = dict(UDP_FLOOD, **{"frame.time": " 2021 22:27:10.1 "})
        with self.assertRaises(ValueError):
            R.classify_layout(frame(bad))


class ValueTests(unittest.TestCase):
    def test_zero_spellings_collapse(self):
        self.assertEqual(R.canonical_value("0.0"), "0")
        self.assertEqual(R.canonical_value(" 0 "), "0")
        self.assertEqual(R.canonical_value("0x00000000"), "0")
        self.assertEqual(R.canonical_value("0x0739"), str(0x0739))
        self.assertEqual(R.canonical_value("MQTT"), "MQTT")
        self.assertTrue(math.isnan(R.canonical_value(float("nan"))))

    def test_spelling_no_longer_separates_classes(self):
        normal = dict(NORMAL_NTP, **{"mqtt.topic": "0"})
        df, prov = R.repair_edgeiiot(frame(normal, TCP_ROW, MITM))
        self.assertEqual(df["mqtt.topic"].tolist(), ["0", "0"])
        self.assertEqual(df["mqtt.conack.flags"].tolist(), ["0", "0"])

    def test_repair_drops_mitm_and_payload_columns(self):
        df, prov = R.repair_edgeiiot(frame(UDP_FLOOD, TCP_ROW, MITM))
        self.assertEqual(len(df), 2)
        self.assertNotIn("dropped_mitm", set(prov))
        for c in R.EDGE_EXTRA_DROP:
            self.assertNotIn(c, df.columns)
        self.assertEqual(list(prov.index), list(df.index))

    def test_infer_numeric(self):
        df = pd.DataFrame({"a": ["0", "12", None], "b": ["0", "GET", "0"]})
        out = R.infer_numeric(df, ["a", "b"])
        self.assertEqual(out["a"].dtype.kind, "f")
        self.assertFalse(pd.api.types.is_numeric_dtype(out["b"]))


class CseTests(unittest.TestCase):
    def test_day_only_columns_dropped(self):
        df = pd.DataFrame(columns=["Dst Port", "Flow ID", "Src IP", "Src Port", "Dst IP", "Label"])
        self.assertEqual(list(R.drop_cse_day_only_columns(df).columns), ["Dst Port", "Label"])


if __name__ == "__main__":
    unittest.main()
