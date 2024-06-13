using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Mail;
using System.Web;

using ProyectoDeteccionBalas.Models;
using MailKit.Security;
using MimeKit.Text;
using MimeKit;
using MailKit.Net.Smtp;

namespace ProyectoDeteccionBalas.Servicios
{
    public class CorreoServicio
    {
        private static string _Host = "smtp.gmail.com";
        private static int _Puerto = 587;

        private static string _NombreEnvia = "SUCAMEC";
        private static string _Correo = "pm0518473@gmail.com";
        private static string _Clave = "nsbn jour nkgk zgdb";

        public static bool Enviar(Correo correodto)
        {
            try
            {
                var email = new MimeMessage();

                email.From.Add(new MailboxAddress(_NombreEnvia, _Correo));
                email.To.Add(MailboxAddress.Parse(correodto.Para));
                email.Subject = correodto.Asunto;
                email.Body = new TextPart(TextFormat.Html)
                {
                    Text = correodto.Contenido
                };

                var smtp = new MailKit.Net.Smtp.SmtpClient();
                smtp.Connect(_Host, _Puerto, SecureSocketOptions.StartTls);

                smtp.Authenticate(_Correo, _Clave);
                smtp.Send(email);
                smtp.Disconnect(true);
                return true;
            }
            catch
            {
                return false;
            }
        }

    }
}