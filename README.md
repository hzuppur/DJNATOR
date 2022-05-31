# DJNATOR - Auto DJ

![ai will take your jobs](djnator.png)

Project by: Hain Zuppur and Tõnis Hendrik Hlebnikov

## Metadata format
```
{
        "file_name": file_name,
        "title": title,
        "artist": artist,
        "album": album,
        "general_bpm": bpm,
        "section_count": number of song_sections,
        "song_sections": [
                {
                "start": start sample,
                "end": end sample,
                "bpm": section_tempo,
                "section_chroma": section_chroma,
                "section_bass": section_bass,
                "section_mids": section_mids,
                "section_highs": section_highs
                "beat_present": boolean
                },
                ...
        ]
}
```