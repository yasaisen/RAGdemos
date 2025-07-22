"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506060435
"""


def prompt_tem(
    context, 
    query, 
):
    return f"""
References:
{context}
Question:
{query.question}
Do not use markdown syntax to answer.
"""

def query_tem(
    query, 
):
    return f"""
{query.question}
"""

def lm_tem(
    text: str, 
):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that answering all questions about UC Berkeley EECS."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }
    ]












