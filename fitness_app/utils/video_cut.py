import json
import subprocess
import os
from pathlib import Path
import ast
from fractions import Fraction
from decimal import Decimal

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

def get_video_metadata(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    command_result = run(cmd)
    return json.loads(command_result.stdout)


def parse_fps(video_metadata):
    vstream = [s for s in video_metadata["streams"] if s.get("codec_type") == "video"][0]

    def frac(video_stream):
        num, den = video_stream.split("/")
        num_int, den_int = int(num), int(den)
        if den_int == 0 or num_int == 0:
            return Fraction(0, 1)
        return Fraction(num_int, den_int)

    fps = frac(vstream.get("avg_frame_rate", "0/1"))
    if fps == 0:
        fps = frac(vstream.get("r_frame_rate", "0/1"))
    return fps


def audio_check(video_metadata):
    return any(s.get("codec_type") == "audio" for s in video_metadata.get("streams", []))


def frac_to_sec_str(frac) -> str:
    q = (Decimal(frac.numerator) / Decimal(frac.denominator)).quantize(Decimal(1).scaleb(-9))
    return format(q, "f")


def cut_one_segment(task):
    (video_path, output_path, start_frame, end_frame, fps, crf, preset, has_audio) = task
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i", video_path,
        "-vf", f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", "auto",
    ]
    if has_audio:
        start_time = Fraction(start_frame, 1) / fps
        end_time = Fraction(end_frame, 1) / fps
        af = f"atrim=start={frac_to_sec_str(start_time)}:end={frac_to_sec_str(end_time)},asetpts=PTS-STARTPTS"
        cmd += [
            "-af", af,
            "-c:a", "aac",
            "-b:a", "192k",
            "-ac", "2",
            "-ar", "48000",
        ]
    else:
        cmd += ["-an"]
    cmd += [output_path]

    run(cmd)

    return output_path


def multi_rep_processing(args):
    (video_path, repetitions, output_dir, fps, crf, preset, has_audio) = args
    number_of_reps = len(repetitions)
    parts = []
    parts.append("[0:v]split={}".format(number_of_reps) + "".join(f"[v{i}]" for i in range(number_of_reps)) + ";")

    rep_vid_labels = []
    for rep in repetitions:
        video_in = f"[v{rep['rep_id'] - 1}]"
        video_out = f"[v{rep['rep_id'] - 1}o]"
        parts.append(
            f"{video_in}trim=start_frame={rep['start_frame']}:end_frame={rep['end_frame']},"
            f"setpts=PTS-STARTPTS{video_out};"
        )
        rep_vid_labels.append(video_out)

    if has_audio:
        rep_aud_labels = []
        parts.append(
            "[0:a]asplit={}".format(number_of_reps) + "".join(f"[a{i}]" for i in range(number_of_reps)) + ";")

        for rep in repetitions:
            start_time = Fraction(rep["start_frame"], 1) / fps
            end_time = Fraction(rep["end_frame"], 1) / fps
            audio_in = f"[a{rep['rep_id'] - 1}]"
            audio_out = f"[a{rep['rep_id'] - 1}o]"

            parts.append(
                f"{audio_in}atrim=start={frac_to_sec_str(start_time)}:"
                f"end={frac_to_sec_str(end_time)},"
                f"asetpts=PTS-STARTPTS{audio_out};"
            )
            rep_aud_labels.append(audio_out)
    filter_complex = " ".join(parts)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", "auto"
    ]

    if has_audio:
        cmd += [
            "-c:a", "aac",
            "-b:a", "160k",
            "-ac", "2",
            "-ar", "48000",
        ]
    outputs = []

    filename = Path(video_path).stem
    for i, (video_label, rep) in enumerate(zip(rep_vid_labels, repetitions)):
        output_path = os.path.join(output_dir, f"{filename}_rep_{rep['rep_id']}.mp4")
        outputs.append(output_path)

        cmd += ["-map", video_label]
        if has_audio:
            cmd += ['-map', rep_aud_labels[i]]
        cmd += [output_path]
    return cmd, outputs


def cut_video_segments(video_path, repetitions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    crf = 20
    preset = "medium"

    video_metadata = get_video_metadata(video_path)
    fps = parse_fps(video_metadata)
    has_audio = audio_check(video_metadata)

    cmd, outputs = multi_rep_processing((video_path, repetitions, output_dir, fps, crf, preset, has_audio))
    run(cmd)
    return True