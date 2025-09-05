#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatLogger v1.3 - 聊天记录管理器
用于记录和管理项目开发过程中的聊天记录、决策和重要信息
"""

import os
import json
import datetime
from pathlib import Path
import hashlib

class ChatLogger:
    """聊天记录管理器"""
    
    def __init__(self, log_dir="chat_logs"):
        """
        初始化ChatLogger
        
        Args:
            log_dir (str): 日志存储目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建日志文件路径
        self.session_file = self.log_dir / "current_session.json"
        self.history_file = self.log_dir / "chat_history.json"
        self.summary_file = self.log_dir / "project_summary.md"
        
        # 初始化会话
        self.current_session = {
            "session_id": self._generate_session_id(),
            "start_time": datetime.datetime.now().isoformat(),
            "messages": [],
            "decisions": [],
            "tasks": [],
            "files_created": [],
            "files_modified": [],
            "errors": []
        }
    
    def _generate_session_id(self):
        """生成会话ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def log_message(self, role, content, message_type="normal"):
        """
        记录消息
        
        Args:
            role (str): 角色 (user, assistant, system)
            content (str): 消息内容
            message_type (str): 消息类型 (normal, error, warning, success)
        """
        message = {
            "timestamp": datetime.datetime.now().isoformat(),
            "role": role,
            "content": content,
            "type": message_type
        }
        
        self.current_session["messages"].append(message)
        print(f"[{message['timestamp']}] {role.upper()}: {content}")
    
    def log_decision(self, decision, reason="", impact="medium"):
        """
        记录决策
        
        Args:
            decision (str): 决策内容
            reason (str): 决策原因
            impact (str): 影响程度 (low, medium, high)
        """
        decision_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "decision": decision,
            "reason": reason,
            "impact": impact
        }
        
        self.current_session["decisions"].append(decision_record)
        print(f"[DECISION] {decision}")
    
    def log_task(self, task, status="pending", priority="medium"):
        """
        记录任务
        
        Args:
            task (str): 任务描述
            status (str): 任务状态 (pending, in_progress, completed, cancelled)
            priority (str): 优先级 (low, medium, high)
        """
        task_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "task": task,
            "status": status,
            "priority": priority
        }
        
        self.current_session["tasks"].append(task_record)
        print(f"[TASK] {task} - {status}")
    
    def log_file_operation(self, operation, file_path, description=""):
        """
        记录文件操作
        
        Args:
            operation (str): 操作类型 (created, modified, deleted)
            file_path (str): 文件路径
            description (str): 操作描述
        """
        file_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": operation,
            "file_path": file_path,
            "description": description
        }
        
        if operation == "created":
            self.current_session["files_created"].append(file_record)
        elif operation == "modified":
            self.current_session["files_modified"].append(file_record)
        
        print(f"[FILE] {operation.upper()}: {file_path}")
    
    def log_error(self, error, context=""):
        """
        记录错误
        
        Args:
            error (str): 错误信息
            context (str): 错误上下文
        """
        error_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "error": error,
            "context": context
        }
        
        self.current_session["errors"].append(error_record)
        print(f"[ERROR] {error}")
    
    def save_session(self):
        """保存当前会话"""
        try:
            # 保存当前会话
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, ensure_ascii=False, indent=2)
            
            # 更新历史记录
            self._update_history()
            
            print(f"会话已保存: {self.session_file}")
            return True
        except Exception as e:
            print(f"保存会话失败: {e}")
            return False
    
    def _update_history(self):
        """更新历史记录"""
        try:
            # 读取现有历史记录
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = {"sessions": []}
            
            # 添加当前会话
            history["sessions"].append(self.current_session)
            
            # 保存历史记录
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"更新历史记录失败: {e}")
    
    def generate_summary(self):
        """生成项目摘要"""
        try:
            summary = []
            summary.append("# 项目开发摘要")
            summary.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary.append("")
            
            # 会话信息
            summary.append("## 当前会话信息")
            summary.append(f"- 会话ID: {self.current_session['session_id']}")
            summary.append(f"- 开始时间: {self.current_session['start_time']}")
            summary.append(f"- 消息数量: {len(self.current_session['messages'])}")
            summary.append("")
            
            # 决策记录
            if self.current_session["decisions"]:
                summary.append("## 重要决策")
                for decision in self.current_session["decisions"]:
                    summary.append(f"- **{decision['decision']}**")
                    summary.append(f"  - 原因: {decision['reason']}")
                    summary.append(f"  - 影响: {decision['impact']}")
                    summary.append(f"  - 时间: {decision['timestamp']}")
                    summary.append("")
            
            # 任务记录
            if self.current_session["tasks"]:
                summary.append("## 任务记录")
                for task in self.current_session["tasks"]:
                    summary.append(f"- **{task['task']}** - {task['status']}")
                    summary.append(f"  - 优先级: {task['priority']}")
                    summary.append(f"  - 时间: {task['timestamp']}")
                    summary.append("")
            
            # 文件操作记录
            if self.current_session["files_created"]:
                summary.append("## 创建的文件")
                for file_op in self.current_session["files_created"]:
                    summary.append(f"- {file_op['file_path']}")
                    if file_op['description']:
                        summary.append(f"  - 描述: {file_op['description']}")
                    summary.append(f"  - 时间: {file_op['timestamp']}")
                    summary.append("")
            
            if self.current_session["files_modified"]:
                summary.append("## 修改的文件")
                for file_op in self.current_session["files_modified"]:
                    summary.append(f"- {file_op['file_path']}")
                    if file_op['description']:
                        summary.append(f"  - 描述: {file_op['description']}")
                    summary.append(f"  - 时间: {file_op['timestamp']}")
                    summary.append("")
            
            # 错误记录
            if self.current_session["errors"]:
                summary.append("## 错误记录")
                for error in self.current_session["errors"]:
                    summary.append(f"- **{error['error']}**")
                    if error['context']:
                        summary.append(f"  - 上下文: {error['context']}")
                    summary.append(f"  - 时间: {error['timestamp']}")
                    summary.append("")
            
            # 保存摘要
            summary_text = "\n".join(summary)
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            
            print(f"项目摘要已生成: {self.summary_file}")
            return summary_text
            
        except Exception as e:
            print(f"生成摘要失败: {e}")
            return None
    
    def load_session(self, session_id=None):
        """加载会话"""
        try:
            if session_id:
                # 从历史记录中加载指定会话
                if self.history_file.exists():
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                    
                    for session in history["sessions"]:
                        if session["session_id"] == session_id:
                            self.current_session = session
                            print(f"已加载会话: {session_id}")
                            return True
            else:
                # 加载当前会话
                if self.session_file.exists():
                    with open(self.session_file, 'r', encoding='utf-8') as f:
                        self.current_session = json.load(f)
                    print("已加载当前会话")
                    return True
            
            print("未找到指定会话")
            return False
            
        except Exception as e:
            print(f"加载会话失败: {e}")
            return False
    
    def get_session_stats(self):
        """获取会话统计信息"""
        stats = {
            "session_id": self.current_session["session_id"],
            "start_time": self.current_session["start_time"],
            "message_count": len(self.current_session["messages"]),
            "decision_count": len(self.current_session["decisions"]),
            "task_count": len(self.current_session["tasks"]),
            "files_created_count": len(self.current_session["files_created"]),
            "files_modified_count": len(self.current_session["files_modified"]),
            "error_count": len(self.current_session["errors"])
        }
        return stats
    
    def export_session(self, format="json"):
        """导出会话数据"""
        try:
            if format == "json":
                export_file = self.log_dir / f"export_{self.current_session['session_id']}.json"
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(self.current_session, f, ensure_ascii=False, indent=2)
                print(f"会话已导出: {export_file}")
                return str(export_file)
            elif format == "markdown":
                summary = self.generate_summary()
                export_file = self.log_dir / f"export_{self.current_session['session_id']}.md"
                with open(export_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"会话已导出: {export_file}")
                return str(export_file)
            else:
                print(f"不支持的导出格式: {format}")
                return None
        except Exception as e:
            print(f"导出会话失败: {e}")
            return None

def main():
    """主函数 - 演示ChatLogger的使用"""
    print("ChatLogger v1.3 - 聊天记录管理器")
    print("=" * 50)
    
    # 创建ChatLogger实例
    logger = ChatLogger()
    
    # 记录一些示例数据
    logger.log_message("user", "开始项目开发", "normal")
    logger.log_message("assistant", "好的，我来帮您开发项目", "normal")
    
    logger.log_decision("使用Python开发", "Python具有丰富的库和社区支持", "high")
    logger.log_decision("采用模块化设计", "便于维护和扩展", "medium")
    
    logger.log_task("创建项目结构", "completed", "high")
    logger.log_task("实现核心功能", "in_progress", "high")
    logger.log_task("编写文档", "pending", "medium")
    
    logger.log_file_operation("created", "main.py", "主程序文件")
    logger.log_file_operation("created", "config.py", "配置文件")
    logger.log_file_operation("modified", "README.md", "更新项目说明")
    
    logger.log_error("模块导入失败", "缺少依赖包")
    
    # 保存会话
    logger.save_session()
    
    # 生成摘要
    logger.generate_summary()
    
    # 显示统计信息
    stats = logger.get_session_stats()
    print("\n会话统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 导出会话
    logger.export_session("json")
    logger.export_session("markdown")
    
    print("\nChatLogger演示完成！")

if __name__ == "__main__":
    main()
